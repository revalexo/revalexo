# metrics/classification_report.py

import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import os
from typing import Dict, Union, Optional, List
from collections import Counter

class ClassificationReport:
    """Classification report metric using sklearn's classification_report with additional count information."""
    
    def __init__(self, output_format: str = "dict", include_csv: bool = True, zero_division: int = 0):
        """
        Initialize the classification report metric.
        
        Args:
            output_format (str): Format of the output ('dict' or 'string')
            include_csv (bool): Whether to return a formatted DataFrame for CSV export
            zero_division (int): Value to use when dividing by zero (0 or 1)
        """
        self.output_format = output_format
        self.include_csv = include_csv
        self.class_names = None
        self.zero_division = zero_division
    
    def __call__(self, outputs: np.ndarray, targets: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Compute classification report.
        
        Args:
            outputs (np.ndarray): Model predictions (N, C) where C is the number of classes
            targets (np.ndarray): Ground truth labels, can be (N,) class indices or (N, C) one-hot encoded targets
            class_names (Optional[List[str]]): Names of the classes
            
        Returns:
            Dict[str, Union[float, pd.DataFrame]]: Dictionary with classification report metrics and optional DataFrame
        """
        # Check if targets are one-hot encoded
        if targets.ndim > 1 and targets.shape[1] > 1:
            # Convert one-hot to class indices
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets.flatten()
            
        # Get predicted classes
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            # Binary classification with single output node
            predicted = (outputs > 0.5).astype(int)
        else:
            # Multi-class case
            predicted = np.argmax(outputs, axis=1)
        
        # Store class names
        if class_names is not None:
            self.class_names = class_names
        
        # Generate labels list for classification report
        all_labels = np.unique(np.concatenate([true_classes, predicted]))
        target_names = None
        if self.class_names is not None:
            target_names = [self.class_names[i] if i < len(self.class_names) else f"Class {i}" 
                           for i in all_labels]
        
        # Generate classification report
        report = classification_report(
            true_classes, 
            predicted,
            labels=all_labels,
            target_names=target_names,
            output_dict=(self.output_format == 'dict'),
            zero_division=self.zero_division
        )
        
        # Convert report to DataFrame for CSV export if requested
        if self.include_csv and self.output_format == 'dict':
            # Count ground truth and predicted instances for each class
            y_true_counts = Counter(true_classes)
            y_pred_counts = Counter(predicted)
            
            # Create DataFrame from report
            df = pd.DataFrame(report).T
            
            # Add index as column for easier CSV reading
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'class'}, inplace=True)
            
            # Add total predicted and ground truth columns
            df['total_predicted'] = 0
            df['total_ground_truth'] = 0
            
            # Fill in counts for each class
            for i, row in df.iterrows():
                class_name = row['class']
                # Skip accuracy, macro avg, weighted avg rows
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                # Find the class index based on the class name
                class_idx = None
                if self.class_names is not None:
                    try:
                        class_idx = self.class_names.index(class_name)
                    except ValueError:
                        # Handle case where class name isn't in self.class_names
                        # This might happen for dynamically created classes
                        pass
                
                # If we couldn't find by name, try "Class X" format or bare integer string
                if class_idx is None and class_name.startswith('Class '):
                    try:
                        class_idx = int(class_name.split(' ')[1])
                    except (IndexError, ValueError):
                        pass

                if class_idx is None:
                    try:
                        class_idx = int(class_name)
                    except ValueError:
                        continue
                
                # Fill in counts if we found the class index
                if class_idx is not None:
                    df.at[i, 'total_predicted'] = y_pred_counts.get(class_idx, 0)
                    df.at[i, 'total_ground_truth'] = y_true_counts.get(class_idx, 0)
            
            # Return both the full report and the DataFrame
            return {
                'report': report,
                'dataframe': df
            }
        
        return {'report': report}
    
    def save_csv(self, df: pd.DataFrame, filepath: str) -> str:
        """
        Save classification report DataFrame to CSV.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Path to save the CSV file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath