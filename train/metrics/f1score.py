# metrics/f1score.py

import numpy as np
from sklearn.metrics import f1_score
from typing import Dict, Union
from sklearn.metrics import precision_score, recall_score

class F1Score:
    """F1 score metric for classification tasks with precision and recall."""
    
    def __init__(self, average: str = 'weighted', include_precision_recall: bool = False):
        """
        Initialize the F1 score metric.
        
        Args:
            average (str): Averaging method for F1 score
                Options: 'micro', 'macro', 'weighted', 'samples', None
            include_precision_recall (bool): Whether to include precision and recall in results
        """
        if average not in ['micro', 'macro', 'weighted', 'samples', None, 'binary']:
            raise ValueError(
                f"Invalid average parameter: {average}. "
                "Must be one of: 'micro', 'macro', 'weighted', 'samples', 'binary', None"
        )
        self.average = average
        self.include_precision_recall = include_precision_recall
    
    def __call__(self, outputs: np.ndarray, targets: np.ndarray) -> Union[float, Dict[str, float]]:
        """
        Compute F1 score and optionally precision and recall.
        
        Args:
            outputs (np.ndarray): Model predictions (N, C) where C is the number of classes
            targets (np.ndarray): Ground truth labels, can be (N,) class indices or (N, C) one-hot encoded targets
            
        Returns:
            Union[float, Dict[str, float]]: F1 score value or dictionary with multiple metrics
        """
        
        # Check if targets are one-hot encoded
        if targets.ndim > 1 and targets.shape[1] > 1:
            # Convert one-hot to class indices
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets
            
        # Handle outputs based on shape
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            # Binary classification with single output node
            predicted = (outputs > 0.5).astype(int)
        else:
            # Multi-class case
            predicted = np.argmax(outputs, axis=1)
        
        # Calculate F1 score
        f1 = f1_score(true_classes, predicted, average=self.average, zero_division=0)
        
        if not self.include_precision_recall:
            return f1
        else:
            # Calculate precision and recall
            precision = precision_score(true_classes, predicted, average=self.average, zero_division=0)
            recall = recall_score(true_classes, predicted, average=self.average, zero_division=0)
            
            return {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }