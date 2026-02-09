# src/utils/evaluation.py
"""
Evaluation utilities for IMU2CLIP contrastive learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import average_precision_score


def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    key_embeds: torch.Tensor,
    recall_ks: tuple = (1, 5, 10)
) -> Dict[str, float]:
    """
    Compute retrieval metrics for contrastive learning.
    
    Args:
        query_embeds: Query embeddings [B, D]
        key_embeds: Key embeddings [B, D]
        recall_ks: k values for Recall@k metrics
        
    Returns:
        Dictionary of retrieval metrics
    """
    batch_size = query_embeds.shape[0]
    device = query_embeds.device
    
    # Normalize embeddings
    query_embeds = F.normalize(query_embeds, p=2, dim=1)
    key_embeds = F.normalize(key_embeds, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(query_embeds, key_embeds.T)
    
    # Get ground truth (diagonal elements are positive pairs)
    ground_truth = torch.arange(batch_size, device=device)
    
    metrics = {}
    
    # Compute Recall@k for each k
    for k in recall_ks:
        k = min(k, batch_size)
        _, top_k_indices = similarity.topk(k=k, dim=1)
        
        # Check if ground truth is in top-k
        correct = (top_k_indices == ground_truth.unsqueeze(1)).any(dim=1).float()
        recall_at_k = correct.mean().item()
        
        metrics[f'R@{k}'] = recall_at_k
    
    # Compute Mean Reciprocal Rank (MRR)
    _, sorted_indices = similarity.sort(dim=1, descending=True)
    ranks = (sorted_indices == ground_truth.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    mrr = (1.0 / ranks.float()).mean().item()
    metrics['MRR'] = mrr
    
    # Compute Mean Average Precision (mAP)
    # For each query, compute AP
    ap_scores = []
    for i in range(batch_size):
        # Create binary relevance vector (1 for correct match, 0 otherwise)
        relevance = torch.zeros(batch_size, device=device)
        relevance[i] = 1
        
        # Get similarity scores for this query
        scores = similarity[i].cpu().numpy()
        relevance_np = relevance.cpu().numpy()
        
        # Compute average precision
        ap = average_precision_score(relevance_np, scores)
        ap_scores.append(ap)
    
    metrics['mAP'] = np.mean(ap_scores)
    
    return metrics


def evaluate_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    modality_1: str = 'imu',
    modality_2: str = 'visual'
) -> Dict[str, float]:
    """
    Evaluate a model on retrieval metrics.
    
    Args:
        model: IMU2CLIP model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        modality_1: Name of first modality
        modality_2: Name of second modality
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_embeds_1 = []
    all_embeds_2 = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (assuming format: imu, visual, labels, transition_flags)
            data_1, data_2, _, _ = batch
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            
            # Get embeddings
            embeddings = model(data_1, data_2)
            
            all_embeds_1.append(embeddings[modality_1])
            all_embeds_2.append(embeddings[modality_2])
    
    # Concatenate all embeddings
    all_embeds_1 = torch.cat(all_embeds_1, dim=0)
    all_embeds_2 = torch.cat(all_embeds_2, dim=0)
    
    # Compute metrics for both directions
    metrics = {}
    
    # Modality 1 -> Modality 2
    metrics_12 = compute_retrieval_metrics(all_embeds_1, all_embeds_2)
    for key, value in metrics_12.items():
        metrics[f'{modality_1}2{modality_2}_{key}'] = value
    
    # Modality 2 -> Modality 1
    metrics_21 = compute_retrieval_metrics(all_embeds_2, all_embeds_1)
    for key, value in metrics_21.items():
        metrics[f'{modality_2}2{modality_1}_{key}'] = value
    
    # Average metrics
    for key in ['R@1', 'R@5', 'R@10', 'MRR', 'mAP']:
        avg_value = (metrics[f'{modality_1}2{modality_2}_{key}'] + 
                     metrics[f'{modality_2}2{modality_1}_{key}']) / 2
        metrics[f'avg_{key}'] = avg_value
    
    return metrics


def compute_confusion_matrix_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrix and per-class metrics.
    
    Args:
        predictions: Predicted class indices [N]
        targets: Target class indices [N]
        num_classes: Number of classes
        
    Returns:
        Dictionary with confusion matrix and per-class metrics
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(range(num_classes)), average=None
    )
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = 'tsne',
    perplexity: int = 30,
    n_components: int = 2
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Visualize embeddings using dimensionality reduction.
    
    Args:
        embeddings: Embeddings to visualize [N, D]
        labels: Optional labels for coloring [N]
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity for t-SNE
        n_components: Number of components to reduce to
        
    Returns:
        Reduced embeddings and optional figure
    """
    embeddings = embeddings.cpu().numpy()
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    
    # Optional: Create visualization
    figure = None
    if labels is not None:
        import matplotlib.pyplot as plt
        
        labels = labels.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
        ax.set_title(f'{method.upper()} Visualization of Embeddings')
        plt.colorbar(scatter, ax=ax)
        figure = fig
    
    return reduced, figure