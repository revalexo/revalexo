# models/evi_mae_fusion.py
"""
Complete EVI-MAE Fusion Model Implementation
Includes all components needed for fine-tuning with pretrained EVI-MAE weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from functools import partial
from .base_models import BaseEncoder, MultiHorizonClassifier

# Import timm components
from timm.models.layers import to_2tuple, trunc_normal_, DropPath, drop_path
from timm.models.vision_transformer import Attention, Mlp

# DGL imports for graph processing
# Install DGL for using (ATM requires torch<=12.4, does not support blackwell GPUs)
try:
    import dgl
    import dgl.function as fn
    from dgl.nn.pytorch.glob import AvgPooling
    from dgl.utils import expand_as_pair
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("WARNING: DGL not installed. Graph features will be disabled.")
    print("Install with: pip install dgl")


# ============================================================================
# Helper Functions
# ============================================================================

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): # num_of_patches, embed_dim
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy
    import numpy as np
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

# ============================================================================
# Patch Embedding Modules
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_video(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],  patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# ============================================================================
# Transformer Blocks
# ============================================================================

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

class Video_DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Video_Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None: # attn_head_dim is None
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias: # here
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Video_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Video_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Video_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Video_DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Video_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else: # here
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None: # here
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Graph Neural Network Components
# ============================================================================

class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func

        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            # aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn = fn.copy_u('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:
            # input projection (no residual)
            self.layers.append(GINConv(
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
            # output projection
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        # print('h0 shape', h.shape) # 630 271
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            # print('hb{} shape'.format(l), h.shape) # 630 512
            hidden_list.append(h)
        # output projection
        if return_hidden:
            # print('return shape', self.head(h).shape) # 630 512
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, 
                activation, residual, norm, nhead, nhead_out, attn_drop, 
                negative_slope=0.2, concat_out=True):
    if m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError(f"Module type {m_type} not implemented")
    
    return mod

# ============================================================================
# EVI-MAE Encoder (inheriting from BaseEncoder)
# ============================================================================

class EVI_MAE_Encoder(BaseEncoder):
    """
    EVI-MAE Encoder that inherits from BaseEncoder.
    This is the exact EVIMAEFT architecture from the original implementation.
    """
    def __init__(
        self,
        video_model_dict: Dict[str, Any],
        imu_model_dict: Dict[str, Any],
        feature_dim: int = 768,
        pretrained_checkpoint: Optional[str] = None,
        freeze_weights: bool = False
    ):
        super().__init__(feature_dim)
        
        # Override timm package
        import timm
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        # Video encoder parameters
        self.video_img_size = video_model_dict['img_size']
        self.video_patch_size = video_model_dict['patch_size']
        self.video_in_chans = 3
        self.video_encoder_embed_dim = video_model_dict['encoder_embed_dim']
        self.video_tubelet_size = 2
        self.video_drop_path_rate = 0.0
        self.video_encoder_depth = video_model_dict['encoder_depth']
        self.video_encoder_num_heads = video_model_dict['encoder_num_heads']
        self.video_mlp_ratio = video_model_dict['mlp_ratio']
        self.video_qkv_bias = video_model_dict['qkv_bias']
        self.video_qk_scale = None
        self.video_drop_rate = 0.0
        self.video_attn_drop_rate = 0.0
        self.video_norm_layer = nn.LayerNorm
        self.video_init_values = 0.0

        # IMU encoder parameters
        self.imu_patch_size = imu_model_dict['patch_size']
        self.imu_channel_num = imu_model_dict['channel_num']
        self.imu_plot_height = imu_model_dict['plot_height']
        self.imu_plot_length = imu_model_dict['target_length']
        self.imu_encoder_num_heads = imu_model_dict['encoder_num_heads']
        self.imu_encoder_depth = imu_model_dict['encoder_depth']
        self.imu_encoder_embed_dim = imu_model_dict['encoder_embed_dim']
        self.video_imu_mlp_ratio = self.video_mlp_ratio
        self.video_imu_qkv_bias = self.video_qkv_bias
        self.video_imu_qk_scale = self.video_qk_scale
        self.video_imu_norm_layer = self.video_norm_layer
        self.imu_enable_graph = imu_model_dict['enable_graph']
        self.imu_two_stream = imu_model_dict.get('imu_two_stream', False)

        # Unified branch
        self.unified_num_heads = self.video_encoder_num_heads
        self.unified_depth = 1
        self.unified_embed_dim = self.video_encoder_embed_dim

        # Patch embeddings
        self.patch_embed_a = PatchEmbed(224, self.imu_patch_size, 3, self.imu_encoder_embed_dim)
        self.patch_embed_video = PatchEmbed_video(
            img_size=self.video_img_size, 
            patch_size=self.video_patch_size,
            in_chans=self.video_in_chans, 
            embed_dim=self.video_encoder_embed_dim, 
            tubelet_size=self.video_tubelet_size
        )
        
        self.imu_patch_width_num = int(self.imu_plot_length / self.imu_patch_size)
        self.imu_patch_height_num = int(self.imu_plot_height / self.imu_patch_size)
        self.patch_embed_a.num_patches = int(self.imu_patch_width_num * self.imu_patch_height_num)

        # Modality embeddings
        self.modality_a = nn.Parameter(torch.zeros(1, 1, self.imu_encoder_embed_dim))
        self.modality_video = nn.Parameter(torch.zeros(1, 1, self.video_encoder_embed_dim))

        # Position embeddings
        self.pos_embed_a = get_sinusoid_encoding_table(self.patch_embed_a.num_patches, self.imu_encoder_embed_dim)
        self.pos_embed_video = get_sinusoid_encoding_table(self.patch_embed_video.num_patches, self.video_encoder_embed_dim)

        # IMU encoder blocks
        self.blocks_a = nn.ModuleList([
            Block(
                self.imu_encoder_embed_dim, self.imu_encoder_num_heads, self.video_imu_mlp_ratio,
                qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale,
                norm_layer=self.video_imu_norm_layer
            ) for i in range(self.imu_encoder_depth)
        ])

        # Video encoder blocks
        dpr = [x.item() for x in torch.linspace(0, self.video_drop_path_rate, self.video_encoder_depth)]
        self.blocks_video = nn.ModuleList([
            Video_Block(
                dim=self.video_encoder_embed_dim, num_heads=self.video_encoder_num_heads,
                mlp_ratio=self.video_imu_mlp_ratio, qkv_bias=self.video_imu_qkv_bias,
                qk_scale=self.video_imu_qk_scale, drop=self.video_drop_rate,
                attn_drop=self.video_attn_drop_rate, drop_path=dpr[i],
                norm_layer=self.video_imu_norm_layer, init_values=self.video_init_values
            ) for i in range(self.video_encoder_depth)
        ])

        # Unified encoder blocks
        self.blocks_u = nn.ModuleList([
            Block(
                self.unified_embed_dim, self.unified_num_heads, self.video_imu_mlp_ratio,
                qkv_bias=self.video_imu_qkv_bias, qk_scale=self.video_imu_qk_scale,
                norm_layer=self.video_imu_norm_layer
            ) for i in range(self.unified_depth)
        ])

        # Normalization layers
        self.norm_a = self.video_imu_norm_layer(self.unified_embed_dim)
        self.norm_video = self.video_imu_norm_layer(self.video_encoder_embed_dim)
        self.norm = self.video_imu_norm_layer(self.unified_embed_dim)

        # Classification head (created BEFORE initialize_weights, matching original)
        if not self.imu_enable_graph or not HAS_DGL:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.unified_embed_dim),
                nn.Linear(self.unified_embed_dim, feature_dim)
            )
        elif self.imu_enable_graph and not self.imu_two_stream:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.unified_embed_dim + 512),
                nn.Linear(self.unified_embed_dim + 512, feature_dim)
            )
        else:  # imu_enable_graph and imu_two_stream
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.unified_embed_dim),
                nn.Linear(self.unified_embed_dim, feature_dim)
            )
            self.another_mlp_head = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, feature_dim)
            )

        # Initialize weights BEFORE graph encoder (matching original order)
        self.initialize_weights()

        # Graph encoder (if enabled) - created AFTER initialize_weights() to match
        # original, so _init_weights is NOT applied to graph parameters
        if self.imu_enable_graph and HAS_DGL:
            g_num_hidden = 512
            g_encoder_type = imu_model_dict['imu_graph_net']
            g_num_layers = 2
            g_nhead = 2
            g_nhead_out = 1
            g_negative_slope = 0.2
            g_in_dim = self.imu_encoder_embed_dim
            g_activation = 'prelu'
            g_feat_drop = 0.2
            g_attn_drop = 0.1
            g_residual = False
            g_norm = 'batchnorm'

            if g_encoder_type in ("gat", "dotgat"):
                g_enc_num_hidden = g_num_hidden // g_nhead
                g_enc_nhead = g_nhead
            else:
                g_enc_num_hidden = g_num_hidden
                g_enc_nhead = 1

            self.graph_encoder = setup_module(
                m_type=g_encoder_type,
                enc_dec="encoding",
                in_dim=g_in_dim,
                num_hidden=g_enc_num_hidden,
                out_dim=g_enc_num_hidden,
                num_layers=g_num_layers,
                nhead=g_enc_nhead,
                nhead_out=g_enc_nhead,
                concat_out=True,
                activation=g_activation,
                dropout=g_feat_drop,
                attn_drop=g_attn_drop,
                negative_slope=g_negative_slope,
                residual=g_residual,
                norm=g_norm,
            )

            self.graph_pooler = AvgPooling()

        # Load pretrained weights if provided
        if pretrained_checkpoint:
            self.load_pretrained_weights(pretrained_checkpoint)

        # Freeze weights if requested
        if freeze_weights:
            self.freeze_encoder_weights()

    def initialize_weights(self):
        """Initialize weights."""
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_video.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_video, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pretrained weights from EVI-MAE pretraining.
        Matches original loading logic: filters by name AND shape compatibility.
        """
        print(f"Loading EVI-MAE pretrained weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove module prefix if present (original uses DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Filter by name AND shape, matching original's loading logic
        model_state_dict = self.state_dict()
        filtered_dict = {}
        skipped_name = []
        skipped_shape = []

        for name, param in state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape == param.shape:
                    filtered_dict[name] = param
                else:
                    skipped_shape.append(f"{name}: checkpoint {param.shape} vs model {model_state_dict[name].shape}")
            else:
                skipped_name.append(name)

        missing, unexpected = self.load_state_dict(filtered_dict, strict=False)

        print(f"Loaded {len(filtered_dict)} pretrained parameters.")
        if skipped_shape:
            print(f"Skipped (shape mismatch): {skipped_shape}")
        if missing:
            print(f"Missing keys (expected for new heads): {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")


    def freeze_encoder_weights(self):
        """Freeze encoder parameters except classification heads."""
        for name, param in self.named_parameters():
            if 'mlp_head' not in name and 'another_mlp_head' not in name:
                param.requires_grad = False

    def encode_features(self, inputs, mode='multimodal'):
        """
        Required by BaseEncoder interface.

        Args:
            inputs: Dictionary with 'raw_imu' and/or 'video' keys
            mode: Forward mode - 'multimodal', 'ft_imuonly', 'ft_videoonly',
                  'inf_imuonly', 'inf_videoonly'

        Returns:
            Feature tensor [B, feature_dim] or tuple of (features, graph_pred)
            when imu_two_stream is True
        """
        if isinstance(inputs, dict):
            raw_imu = inputs.get('raw_imu')
            video = inputs.get('video')
        elif isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            raw_imu = inputs[0]
            video = inputs[1]
        else:
            raise ValueError("Inputs must be dict with 'raw_imu' and 'video' keys or list/tuple")

        # Process IMU format
        if raw_imu is not None:
            if raw_imu.dim() == 5:
                if raw_imu.shape[1] == 1:
                    raw_imu = raw_imu.squeeze(1)
                elif raw_imu.shape[1] == 4 and raw_imu.shape[2] == 3:
                    batch_size = raw_imu.shape[0]
                    raw_imu = raw_imu.view(batch_size, 12, raw_imu.shape[3], raw_imu.shape[4])

        return self.forward_encoder(raw_imu, video, mode=mode)

    def _imu_dim_align(self, a):
        """Align IMU dimension (768) to video dimension (384) via avg_pool1d."""
        if a.shape[2] != self.video_encoder_embed_dim:
            if a.shape[2] == 768 and self.video_encoder_embed_dim == 384:
                a = F.avg_pool1d(a, kernel_size=2, stride=2)
            else:
                raise ValueError(f"Unexpected dimension mismatch: a={a.shape[2]}, video={self.video_encoder_embed_dim}")
        return a

    def _imu_preprocess(self, a):
        """Common IMU preprocessing: split limbs, patch embed, add pos/modality."""
        assert len(a.shape) == 4
        assert a.shape[1] == self.imu_channel_num

        a_left_arm = a[:, 0:3, :, :]
        a_right_arm = a[:, 3:6, :, :]
        a_left_leg = a[:, 6:9, :, :]
        a_right_leg = a[:, 9:12, :, :]
        a = torch.cat((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0)
        bs = int(a.shape[0] / 4)

        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
        a = a + self.modality_a
        return a, bs

    def _graph_process(self, a, bs):
        """Run graph construction and encoding on IMU features."""
        a_for_graph = a.clone()
        for blk in self.blocks_a:
            a_for_graph = blk(a_for_graph)

        a_left_arm_for_graph = torch.mean(a_for_graph[0:bs, :, :], dim=1)
        a_right_arm_for_graph = torch.mean(a_for_graph[bs:2*bs, :, :], dim=1)
        a_left_leg_for_graph = torch.mean(a_for_graph[2*bs:3*bs, :, :], dim=1)
        a_right_leg_for_graph = torch.mean(a_for_graph[3*bs:4*bs, :, :], dim=1)

        u_edge = torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3])
        v_edge = torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
        body_graph = dgl.graph((u_edge, v_edge)).to(a.device)

        body_graphs = []
        for bi in range(bs):
            body_graph_i = body_graph.clone()
            stacked_features = torch.stack((
                a_left_arm_for_graph[bi],
                a_right_arm_for_graph[bi],
                a_left_leg_for_graph[bi],
                a_right_leg_for_graph[bi]
            ), dim=0)
            body_graph_i.ndata['attr'] = stacked_features
            body_graphs.append(body_graph_i)

        body_graphs_batch = dgl.batch(body_graphs)
        body_graphs_batch_feat = body_graphs_batch.ndata["attr"]

        enc_rep, _ = self.graph_encoder(body_graphs_batch, body_graphs_batch_feat, return_hidden=True)
        return self.graph_pooler(body_graphs_batch, enc_rep)

    def _imu_mean_and_encode(self, a, bs):
        """Average limb features and run through IMU encoder blocks."""
        a_left_arm = a[0:bs, :, :]
        a_right_arm = a[bs:2*bs, :, :]
        a_left_leg = a[2*bs:3*bs, :, :]
        a_right_leg = a[3*bs:4*bs, :, :]
        a = torch.mean(torch.stack((a_left_arm, a_right_arm, a_left_leg, a_right_leg), dim=0), dim=0)

        for blk in self.blocks_a:
            a = blk(a)
        return a

    def _graph_output(self, x, graph_enc_rep_Bx512):
        """Apply graph features to output, matching original's return behavior."""
        if self.imu_enable_graph and HAS_DGL and graph_enc_rep_Bx512 is not None:
            if not self.imu_two_stream:
                x = torch.cat((x, graph_enc_rep_Bx512), dim=1)
                x = self.mlp_head(x)
            else:
                x = self.mlp_head(x)
                graph_pred = self.another_mlp_head(graph_enc_rep_Bx512)
                # Return tuple matching original - caller handles combination
                return x, graph_pred
        else:
            x = self.mlp_head(x)
        return x

    def forward_encoder(self, a, v, mode='multimodal'):
        """
        Forward pass with mode selection, matching original EVIMAEFT.forward().

        Args:
            a: IMU spectrogram [B, 12, H, W] (can be None for video-only modes)
            v: Video tensor [B, 3, T, H, W] (can be None for IMU-only modes)
            mode: One of 'multimodal', 'ft_imuonly', 'ft_videoonly',
                  'inf_imuonly', 'inf_videoonly'

        Returns:
            Feature tensor [B, feature_dim] or tuple (features, graph_pred)
        """
        if mode == 'multimodal':
            return self._forward_multimodal(a, v)
        elif mode == 'ft_imuonly':
            return self._forward_ft_imuonly(a)
        elif mode == 'ft_videoonly':
            return self._forward_ft_videoonly(v)
        elif mode == 'inf_imuonly':
            return self._forward_inf_imuonly(a)
        elif mode == 'inf_videoonly':
            return self._forward_inf_videoonly(v)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_multimodal(self, a, v):
        """Multimodal forward pass (matching original mode='multimodal')."""
        a, bs = self._imu_preprocess(a)

        # Video preprocessing
        v = self.patch_embed_video(v)
        v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
        v = v + self.modality_video

        # Graph processing
        graph_enc_rep_Bx512 = None
        if self.imu_enable_graph and HAS_DGL:
            graph_enc_rep_Bx512 = self._graph_process(a, bs)

        # Average limb features and encode
        a = self._imu_mean_and_encode(a, bs)

        for blk in self.blocks_video:
            v = blk(v)

        # Align dimensions (IMU 768 -> 384)
        if a.shape[2] != v.shape[2]:
            a = self._imu_dim_align(a)

        # Concatenate and unified processing
        x = torch.cat((a, v), dim=1)
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return self._graph_output(x, graph_enc_rep_Bx512)

    def _forward_ft_imuonly(self, a):
        """IMU-only finetuning forward (matching original mode='ft_imuonly')."""
        a, bs = self._imu_preprocess(a)

        # Graph processing
        graph_enc_rep_Bx512 = None
        if self.imu_enable_graph and HAS_DGL:
            graph_enc_rep_Bx512 = self._graph_process(a, bs)

        # Average limb features and encode
        a = self._imu_mean_and_encode(a, bs)

        # Align dimension
        a = self._imu_dim_align(a)

        # Unified block with modality-specific normalization ('a')
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)
        x = a.mean(dim=1)

        return self._graph_output(x, graph_enc_rep_Bx512)

    def _forward_ft_videoonly(self, v):
        """Video-only finetuning forward (matching original mode='ft_videoonly')."""
        v = self.patch_embed_video(v)
        v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
        v = v + self.modality_video

        for blk in self.blocks_video:
            v = blk(v)

        # Unified block with modality-specific normalization ('v')
        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_video(v)
        x = v.mean(dim=1)

        x = self.mlp_head(x)
        return x

    def _forward_inf_imuonly(self, a):
        """
        IMU-only inference forward (matching original mode='inf_imuonly').
        Uses dual-path averaging: unified norm + modality-specific norm.
        """
        if len(a.shape) == 3:
            a = a.unsqueeze(1)
        assert len(a.shape) == 4

        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a.type_as(a).to(a.device).clone().detach()
        a = a + self.modality_a

        for blk in self.blocks_a:
            a = blk(a)

        a = self._imu_dim_align(a)

        # Path 1: unified normalization
        u = a
        for blk in self.blocks_u:
            u = blk(u)  # unified normalization (no modality arg)
        u = self.norm(u)
        u = u.mean(dim=1)

        # Path 2: modality-specific normalization
        for blk in self.blocks_u:
            a = blk(a, 'a')  # modality-specific normalization
        a = self.norm_a(a)
        a = a.mean(dim=1)

        # Average the two paths
        x = (u + a) / 2
        x = self.mlp_head(x)
        return x

    def _forward_inf_videoonly(self, v):
        """
        Video-only inference forward (matching original mode='inf_videoonly').
        Uses dual-path averaging: unified norm + modality-specific norm.
        """
        v = self.patch_embed_video(v)
        v = v + self.pos_embed_video.type_as(v).to(v.device).clone().detach()
        v = v + self.modality_video

        for blk in self.blocks_video:
            v = blk(v)

        # Path 1: unified normalization
        u = v
        for blk in self.blocks_u:
            u = blk(u)  # unified normalization
        u = self.norm(u)
        u = u.mean(dim=1)

        # Path 2: modality-specific normalization
        for blk in self.blocks_u:
            v = blk(v, 'v')  # modality-specific normalization
        v = self.norm_video(v)
        v = v.mean(dim=1)

        # Average the two paths
        x = (u + v) / 2
        x = self.mlp_head(x)
        return x

    def get_feature_dim(self):
        """Required by BaseEncoder interface."""
        return self.feature_dim


# ============================================================================
# Final Fusion Model (following your codebase structure)
# ============================================================================

class EVI_MAE_Fusion(nn.Module):
    """
    EVI-MAE Fusion model following AidWear codebase structure.
    Uses EVI_MAE_Encoder with MultiHorizonClassifier.
    """
    def __init__(
        self,
        video_model_dict: Dict[str, Any],
        imu_model_dict: Dict[str, Any],
        num_classes: int = 13,
        prediction_horizons: List[float] = [0],
        feature_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        pretrained_checkpoint: Optional[str] = None,
        freeze_encoders: bool = False,
        shared_classifier_layers: bool = True,
        modalities: List[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.modalities = modalities if modalities else ["raw_imu", "video"]
        self.prediction_horizons = prediction_horizons
        self.num_prediction_heads = len(prediction_horizons)
        
        print("\n" + "="*60)
        print("Initializing EVI-MAE Fusion Model")
        print(f"  Modalities: {self.modalities}")
        print(f"  Prediction horizons: {prediction_horizons}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Feature dimension: {feature_dim}")
        if pretrained_checkpoint:
            print(f"  Pretrained checkpoint: {pretrained_checkpoint}")
        print("="*60 + "\n")

        # Initialize encoder
        self.encoder = EVI_MAE_Encoder(
            video_model_dict=video_model_dict,
            imu_model_dict=imu_model_dict,
            feature_dim=feature_dim,
            pretrained_checkpoint=pretrained_checkpoint,
            freeze_weights=freeze_encoders
        )

        # Multi-horizon classifier (matching your other models)
        self.classifier = MultiHorizonClassifier(
            input_dim=feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            shared_layers=shared_classifier_layers
        )

    def forward(self, inputs=None, raw_imu=None, video=None, mode='multimodal', **kwargs):
        """
        Forward pass matching MuLoMo codebase interface.

        Args:
            inputs: Dict with 'raw_imu'/'video' keys, or list/tuple
            raw_imu: IMU spectrogram [B, 12, H, W]
            video: Video tensor [B, 3, T, H, W]
            mode: Forward mode - 'multimodal', 'ft_imuonly', 'ft_videoonly',
                  'inf_imuonly', 'inf_videoonly'
        """
        # Extract inputs
        if inputs is not None:
            if isinstance(inputs, dict):
                raw_imu = inputs.get('raw_imu', raw_imu)
                video = inputs.get('video', video)
            elif isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
                raw_imu = inputs[0]
                video = inputs[1]

        # Also check kwargs
        raw_imu = kwargs.get('raw_imu', raw_imu)
        video = kwargs.get('video', video)

        # Auto-detect mode based on available inputs
        if mode == 'multimodal' and (raw_imu is None or video is None):
            if raw_imu is not None and video is None:
                mode = 'ft_imuonly'
            elif video is not None and raw_imu is None:
                mode = 'ft_videoonly'
            else:
                raise ValueError("EVI-MAE requires at least one modality input")

        # Encode features using the encoder
        encoder_output = self.encoder.encode_features(
            {'raw_imu': raw_imu, 'video': video}, mode=mode
        )

        # Handle two-stream return (tuple of features and graph_pred)
        if isinstance(encoder_output, tuple):
            features, graph_features = encoder_output
            # Both streams go through the classifier independently
            predictions = self.classifier(features)
            graph_predictions = self.classifier(graph_features)
            # Return both for separate loss computation in training loop
            return predictions, graph_predictions

        # Standard single-stream
        predictions = self.classifier(encoder_output)
        return predictions

    def get_num_prediction_heads(self):
        return self.num_prediction_heads

    def get_prediction_horizons(self):
        return self.prediction_horizons