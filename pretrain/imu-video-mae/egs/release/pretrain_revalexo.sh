#!/bin/bash
# Pretraining script for RevalExo dataset with EVI-MAE
# RevalExo uses 7 IMU sensors (lower body): pelvis, right/left upper leg, lower leg, foot

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
dataset=wear
# Set DATASET_BASE_PATH env var to override, or use default relative path
dataset_base_path=${DATASET_BASE_PATH:-$(cd "$(dirname "$0")/../.." && pwd)/data_release/revalexo-release}

# These paths will be automatically determined based on the migration script output
# The script will find the correct JSON files based on the sample count
train_json_path=$(ls ${dataset_base_path}/cav_label/train_pretrain/labels_*.json | head -1)
test_json_path=$(ls ${dataset_base_path}/cav_label/test_pretrain/labels_*.json | head -1)

# Verify files exist
if [ ! -f "$train_json_path" ]; then
    echo "Error: Training JSON not found. Please run migrate_revalexo_data.py first."
    exit 1
fi

if [ ! -f "$test_json_path" ]; then
    echo "Error: Test JSON not found. Please run migrate_revalexo_data.py first."
    exit 1
fi

data_train=${train_json_path}
data_val=${test_json_path}
label_csv=${dataset_base_path}/cav_label/class_labels_indices.csv
pretrain_path=${dataset_base_path}/../videomae_adapt_ckpt/video_imu_beforepretrain_model_small_only_encoder.pth
exp_dir=${dataset_base_path}/evi-mae-exp/${current_time}-pretrain-revalexo

# Check if checkpoint exists
if [ ! -f "$pretrain_path" ]; then
    echo "Warning: VideoMAE checkpoint not found at $pretrain_path"
    echo "Please download it from: https://drive.google.com/file/d/1JxQtmgoxIxqFdY-3CAjUlZvs2JTn_l3A/view?usp=share_link"
    echo "Or set load_prepretrain=False to train from scratch (not recommended)"
    exit 1
fi

model=evi-mae
batch_size=16
epoch=300
lrscheduler_start=100
lrscheduler_decay=0.5
lrscheduler_step=100
load_prepretrain=True
contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True # if True, no inpainting
pretrain_modality=both # imu, video, or both

# IMU configuration for RevalExo
# 4 body parts x 3 axes = 12 channels (same as AidWear)
# Mapping: left_arm->Left_Foot, right_arm->Right_Foot, left_leg->Left_Upper_Leg, right_leg->Right_Upper_Leg
imu_plot_type=stft
imu_channel_num=12 # 4 body parts x 3 axes
imu_target_length=320 # need to be 16x
imu_plot_height=128 # need to be 16x
imu_patch_size=16

# The script will output the actual mean and std values
# These will be updated from imu_statistics.txt if available
imu_dataset_mean=0.0   # UPDATE THIS with actual mean from migrate_revalexo_data.py output
imu_dataset_std=1.0    # UPDATE THIS with actual std from migrate_revalexo_data.py output

# Check if statistics file exists and source it
if [ -f "${dataset_base_path}/imu_statistics.txt" ]; then
    echo "Loading IMU statistics from file..."
    mean_line=$(grep "Mean:" ${dataset_base_path}/imu_statistics.txt)
    std_line=$(grep "Std:" ${dataset_base_path}/imu_statistics.txt)
    imu_dataset_mean=$(echo $mean_line | cut -d':' -f2 | xargs)
    imu_dataset_std=$(echo $std_line | cut -d':' -f2 | xargs)
    echo "Using IMU mean: $imu_dataset_mean, std: $imu_dataset_std"
fi

imu_masking_ratio=0.75
imu_mask_mode=unstructured # or time, or freq, or tf

imu_enable_graph=True
imu_graph_net=gin
imu_graph_masking_ratio=0.5

# Small model configuration (matching WEAR)
imu_encoder_embed_dim=768
imu_encoder_depth=11
imu_encoder_num_heads=12

# Video configuration
video_img_size=224
video_patch_size=16
video_encoder_num_classes=0
video_decoder_num_classes=1536
video_mlp_ratio=4
video_qkv_bias=True
video_masking_ratio=0.9

video_encoder_embed_dim=384
video_encoder_depth=12
video_encoder_num_heads=6
video_decoder_embed_dim=192
video_decoder_num_heads=3

image_as_video=False

mkdir -p $exp_dir
echo "Experiment directory: $exp_dir"
echo "Training data: $data_train"
echo "Validation data: $data_val"
echo "IMU mean: $imu_dataset_mean, std: $imu_dataset_std"
echo "IMU channels: $imu_channel_num (4 body parts x 3 axes)"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Display GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Note: Adjust num-workers based on your system capabilities
# Reduce batch_size if you encounter OOM errors
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python3 -W ignore ../../src/run_evimae_pretrain.py --model ${model} --dataset ${dataset} \
--data-train ${data_train} --data-val ${data_val} --exp-dir $exp_dir \
--label-csv ${label_csv} \
--n-epochs ${epoch} --batch-size $batch_size --save_model True \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--warmup True \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--imu_plot_type ${imu_plot_type} --imu_plot_height ${imu_plot_height} --imu_patch_size ${imu_patch_size} \
--imu_dataset_mean ${imu_dataset_mean} --imu_dataset_std ${imu_dataset_std} --imu_channel_num ${imu_channel_num} \
--imu_masking_ratio ${imu_masking_ratio} --imu_mask_mode ${imu_mask_mode} --imu_target_length ${imu_target_length} \
--video_img_size ${video_img_size} --video_patch_size ${video_patch_size} --video_encoder_num_classes ${video_encoder_num_classes} \
--video_decoder_num_classes ${video_decoder_num_classes} --video_mlp_ratio ${video_mlp_ratio} --video_qkv_bias ${video_qkv_bias} \
--video_encoder_embed_dim ${video_encoder_embed_dim} --video_encoder_depth ${video_encoder_depth} \
--video_encoder_num_heads ${video_encoder_num_heads} --video_decoder_embed_dim ${video_decoder_embed_dim} \
--video_decoder_num_heads ${video_decoder_num_heads} --video_masking_ratio ${video_masking_ratio} \
--imu_encoder_embed_dim ${imu_encoder_embed_dim} --imu_encoder_depth ${imu_encoder_depth} \
--imu_encoder_num_heads ${imu_encoder_num_heads} --load_prepretrain ${load_prepretrain} \
--imu_enable_graph ${imu_enable_graph} --imu_graph_net ${imu_graph_net} \
--pretrain_modality ${pretrain_modality} --image_as_video ${image_as_video} --imu_graph_masking_ratio ${imu_graph_masking_ratio} \
--num-workers 16
