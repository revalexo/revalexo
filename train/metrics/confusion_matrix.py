# metrics/confusion_matrix.py

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    """Confusion matrix for classification tasks."""

    def __init__(self, normalize: bool = False, include_csv: bool = True):
        """
        Initialize the confusion matrix metric.

        Args:
            normalize (bool): Whether to normalize the confusion matrix
            include_csv (bool): Whether to return a formatted DataFrame for CSV export
        """
        self.normalize = normalize
        self.include_csv = include_csv
        self.class_names = None
    
    def __call__(self, outputs: np.ndarray, targets: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute confusion matrix and derived metrics.
        
        Args:
            outputs (np.ndarray): Model predictions (N, C) where C is the number of classes
            targets (np.ndarray): Ground truth labels (N,) or (N, C) one-hot encoded
            class_names (Optional[List[str]]): Names of the classes
            
        Returns:
            Dict[str, float]: Dictionary of metrics derived from confusion matrix
        """
        # Convert one-hot encoded targets to class indices if needed
        if targets.ndim == 2 and targets.shape[1] > 1:
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets.flatten()
        
        # Handle both binary and multi-class outputs
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            # Binary classification with single output (sigmoid)
            predicted = (outputs.flatten() > 0.5).astype(int)
        else:
            # Multi-class classification
            predicted = np.argmax(outputs, axis=1)
        
        if class_names is not None:
            self.class_names = class_names
        
        # Compute the confusion matrix
        cm = confusion_matrix(true_classes, predicted)
        
        # Store the raw confusion matrix as an attribute for later access
        self.matrix = cm.copy()
        
        if self.normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
            self.normalized_matrix = cm
        
        # Calculate metrics from raw confusion matrix (not normalized)
        metrics = {}

        # Diagonal elements are the correctly classified samples for each class
        metrics['accuracy'] = np.trace(self.matrix) / np.sum(self.matrix)

        # Per-class metrics with proper division handling
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_recall = np.diag(self.matrix) / np.sum(self.matrix, axis=1)
            per_class_precision = np.diag(self.matrix) / np.sum(self.matrix, axis=0)
            
            # Remove NaN values (from 0/0 divisions)
            per_class_recall = np.nan_to_num(per_class_recall)
            per_class_precision = np.nan_to_num(per_class_precision)
        
        # Average metrics
        metrics['macro_recall'] = np.mean(per_class_recall)
        metrics['macro_precision'] = np.mean(per_class_precision)
        
        # Add per-class metrics if not too many classes
        if cm.shape[0] <= 10:  # Only add per-class metrics if fewer than 10 classes
            for i in range(cm.shape[0]):
                class_name = self.class_names[i] if self.class_names else f"class_{i}"
                metrics[f'recall_{class_name}'] = per_class_recall[i]
                metrics[f'precision_{class_name}'] = per_class_precision[i]

        # Create DataFrame for CSV export if requested
        if self.include_csv:
            # Get class labels for rows/columns
            if self.class_names is not None:
                labels = self.class_names[:cm.shape[0]]
            else:
                labels = [f"class_{i}" for i in range(cm.shape[0])]

            # Create confusion matrix DataFrame
            cm_df = pd.DataFrame(
                self.matrix,
                index=labels,
                columns=labels
            )
            cm_df.index.name = 'true_label'
            cm_df.columns.name = 'predicted_label'

            # Reset index to make it a column for easier CSV reading
            cm_df = cm_df.reset_index()

            return {
                'metrics': metrics,
                'dataframe': cm_df
            }

        return metrics
    
    def __str__(self) -> str:
        """Return string representation."""
        if hasattr(self, 'matrix'):
            return f"Confusion Matrix: \n{self.matrix}"
        return "Confusion Matrix (not computed yet)"
    
    def plot(self, save_path: Optional[str] = None):
        """
        Plot the confusion matrix.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if not hasattr(self, 'matrix'):
            print("Confusion matrix not computed yet. Call the object first.")
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            
            # Use normalized matrix if available
            cm_to_plot = self.normalized_matrix if hasattr(self, 'normalized_matrix') else self.matrix
            
            if self.class_names is not None:
                sns.heatmap(cm_to_plot, annot=True, 
                            fmt='.2f' if self.normalize else 'd', 
                            xticklabels=self.class_names, 
                            yticklabels=self.class_names)
            else:
                sns.heatmap(cm_to_plot, annot=True, 
                            fmt='.2f' if self.normalize else 'd')
            
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        
        except ImportError:
            print("Matplotlib and/or seaborn not installed. Cannot plot confusion matrix.")