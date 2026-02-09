# metrics/accuracy.py

import numpy as np

class Accuracy:
    """Accuracy metric for classification tasks."""
    
    def __init__(self, top_k: int = 1):
        """
        Initialize the accuracy metric.
        
        Args:
            top_k (int): Number of top predictions to consider for accuracy
        """
        self.top_k = top_k
    
    def __call__(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute accuracy.
        
        Args:
            outputs (np.ndarray): Model predictions - either:
                - (N,) for binary classification (probabilities or logits)
                - (N, C) for multi-class (probabilities or logits)
            targets (np.ndarray): Ground truth labels - either:
                - (N,) class indices
                - (N, C) one-hot encoded
            
        Returns:
            float: Accuracy value
        """
        # Convert one-hot to class indices if needed
        if targets.ndim == 2:
            # One-hot encoded
            true_classes = np.argmax(targets, axis=1)
        else:
            # Already class indices
            true_classes = targets.flatten()
        
        # Binary classification
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            outputs = outputs.flatten()
            # Expect probabilities - assuming sigmoid already applied
            predicted = (outputs > 0.5).astype(int)
            return np.mean(predicted == true_classes)
        
        # Multi-class classification
        elif outputs.ndim == 2:
            num_classes = outputs.shape[1]
            
            # Validate top_k
            if self.top_k > num_classes:
                raise ValueError(f"top_k={self.top_k} cannot be greater than number of classes ({num_classes})")
            
            if self.top_k == 1:
                # Top-1 accuracy
                predicted = np.argmax(outputs, axis=1)
                correct = np.sum(predicted == true_classes)
            else:
                # Top-k accuracy
                top_k_preds = np.argsort(outputs, axis=1)[:, -self.top_k:]
                correct = np.any(top_k_preds == true_classes[:, None], axis=1).sum()

            
            total = len(true_classes)
            return correct / total if total > 0 else 0.0
        
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")