# utils/data_utils.py

import importlib
from typing import Dict, Any, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import torch
from datasets import *
import random
import yaml
import os

# Not used for now
# TODO: Bug fix 
class WeightedDistributedSampler(DistributedSampler):
    """
    A sampler that supports both distributed sampling and class balancing
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, weights=None):
        super(WeightedDistributedSampler, self).__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.weights = weights
        
    def __iter__(self):
        # If no weights are provided, use standard DistributedSampler behavior
        if self.weights is None:
            return super().__iter__()
        
        # Set the random seed for reproducibility
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(self.weights, self.total_size, replacement=True).tolist()
        else:
            # Without shuffle, just repeat indices weighted by class frequency
            indices = []
            for i, w in enumerate(self.weights):
                # Add index i proportional to its weight
                count = int(w * self.total_size)
                indices.extend([i] * count)
            # Ensure we have exactly total_size indices
            if len(indices) > self.total_size:
                indices = indices[:self.total_size]
            elif len(indices) < self.total_size:
                indices.extend([0] * (self.total_size - len(indices)))
        
        # Subset indices for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Also propagate to dataset if it supports epoch setting.
        
        Args:
            epoch (int): Epoch number.
        """
        super().set_epoch(epoch)
        
        # Propagate epoch to dataset if it has set_epoch method
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

class EpochAwareWeightedSampler(torch.utils.data.Sampler):
    """Weighted sampler that changes sampling pattern each epoch."""
    
    def __init__(self, weights, num_samples, base_seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.base_seed = base_seed
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __iter__(self):
        # New generator each epoch with different seed
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch * 1000)
        
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples



# Helper function to convert lists to tuples for size parameters
def convert_size_params(params):
    """Convert list parameters to tuples for torchvision compatibility."""
    converted = params.copy()
    
    # Parameters that should be tuples, not lists
    tuple_params = ['size', 'crop_size', 'scale', 'ratio']
    
    for param_name in tuple_params:
        if param_name in converted and isinstance(converted[param_name], list):
            converted[param_name] = tuple(converted[param_name])
    
    return converted

def multi_horizon_collate_fn(batch):
    """
    Custom collate function to handle multi-horizon labels, per-horizon transition flags,
    and subject IDs for per-subject evaluation.

    Expected batch structure per sample:
        (modality_data..., labels, transition_flags, subject_id)

    Returns:
        tuple: (modality_tensors..., labels_list, transition_flags_list, subjects_list)
    """
    if len(batch) == 0:
        return []

    # Determine structure from first sample
    first_sample = batch[0]

    if not isinstance(first_sample, (list, tuple)):
        # Single item batch - use default collate
        return torch.utils.data.default_collate(batch)

    # Multi-item batch (modalities + labels + transition_flags + subject_id)
    num_items = len(first_sample)

    # Collect items by position
    collected_items = [[] for _ in range(num_items)]
    for sample in batch:
        for i, item in enumerate(sample):
            collected_items[i].append(item)

    # Process each item
    result = []
    for i, items in enumerate(collected_items):
        if i == num_items - 3:  # Labels (third to last)
            # Check if this is multi-horizon or single-horizon
            first_item = items[0]

            if isinstance(first_item, list) and len(first_item) > 1:
                # Multi-horizon: items is list of lists
                # Convert to list of tensors where each tensor contains labels for all samples for that horizon
                num_horizons = len(first_item)
                horizon_labels = []
                for h in range(num_horizons):
                    h_labels = [sample_labels[h] for sample_labels in items]
                    # Use default collate to create tensors properly - this handles CUDA memory correctly
                    horizon_tensor = torch.utils.data.default_collate(h_labels)
                    horizon_labels.append(horizon_tensor)
                result.append(horizon_labels)
            else:
                # Single horizon case: could be list with one element or direct values
                if isinstance(first_item, list) and len(first_item) == 1:
                    # Single horizon wrapped in list - extract the single values
                    single_labels = [sample_labels[0] for sample_labels in items]
                    # Create a list with one tensor (to maintain consistency with multi-horizon format)
                    single_tensor = torch.utils.data.default_collate(single_labels)
                    result.append([single_tensor])  # Wrap in list for consistency
                else:
                    # Direct label values - use default collate and wrap in list
                    single_tensor = torch.utils.data.default_collate(items)
                    result.append([single_tensor])  # Wrap in list for consistency

        elif i == num_items - 2:  # Transition flags (second to last) - now a list of booleans per horizon
            # Check if this is per-horizon flags or single flag
            first_item = items[0]

            if isinstance(first_item, list):
                # Per-horizon transition flags: items is list of lists
                num_horizons = len(first_item)
                horizon_flags = []
                for h in range(num_horizons):
                    h_flags = [sample_flags[h] for sample_flags in items]
                    # Convert to tensor
                    horizon_tensor = torch.tensor(h_flags, dtype=torch.bool)
                    horizon_flags.append(horizon_tensor)
                result.append(horizon_flags)
            else:
                # Single transition flag (backward compatibility)
                # Convert to tensor and wrap in list for consistency
                flags_tensor = torch.tensor(items, dtype=torch.bool)
                result.append([flags_tensor])  # Wrap in list

        elif i == num_items - 1:  # Subject IDs (last) - list of strings
            # Keep as list of strings for per-subject evaluation
            result.append(items)
        else:
            # Modality data - use default collate
            result.append(torch.utils.data.default_collate(items))

    return tuple(result) if len(result) > 1 else result[0]

def setup_transforms(config: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
    """
    Set up data transforms for each modality based on configuration.
    
    Args:
        config: Configuration dictionary
        split: Data split to set up transforms for ('train' or 'eval')
        
    Returns:
        Dictionary of transform functions for each modality
    """
    transforms = {}
    transform_suffix = 'train_transforms' if split == 'train' else 'eval_transforms'
    
    # Import all transforms from the transforms module
    transform_module = importlib.import_module('transforms')
        
    # Set up transforms for each modality
    for modality, modality_config in config.get('modalities', {}).items():
        transforms_config = modality_config.get(transform_suffix, [])
        
        if not transforms_config:
            # No transforms specified, use identity transform
            transforms[modality] = transform_module.Identity()
            continue
        
        # Create transform list
        transform_list = []
        for transform_info in transforms_config:
            transform_name = transform_info.get('name')
            transform_params = transform_info.get('params', {})
            
            # Convert list parameters to tuples for video transforms
            if modality == 'video':
                transform_params = convert_size_params(transform_params)
            
            # Get transform class from module
            if hasattr(transform_module, transform_name):
                transform_class = getattr(transform_module, transform_name)
                transform = transform_class(**transform_params)
                transform_list.append(transform)
            else:
                raise ValueError(f"Transform {transform_name} not found in transforms module")
        
        # Compose transforms
        if transform_list:
            transforms[modality] = transform_module.Compose(transform_list)
        else:
            transforms[modality] = transform_module.Identity()
    
    return transforms


def setup_datasets_and_loaders(config: Dict[str, Any], distributed: bool = False, 
                              rank: int = 0, world_size: int = 1) -> Dict[str, Union[Dataset, DataLoader]]:
    """
    Set up datasets and data loaders based on configuration.
    Now supports config overrides - anything in the training config's 'dataset' section
    will override the corresponding values in the base dataset config file.
    
    Args:
        config: Configuration dictionary
        distributed: Whether to set up for distributed training
        rank: Current process rank for distributed training
        world_size: Total number of processes for distributed training
        
    Returns:
        Dictionary containing datasets and data loaders
    """
    # Import the dataset class
    dataset_module = importlib.import_module('datasets.multimodaldataset')
    dataset_class = getattr(dataset_module, 'MultimodalSensorDataset')
    
    # Get dataset config
    dataset_name = config['dataset']['name']
    dataset_config_path = config['dataset']['config_path']

    # Get base seed for reproducibility (should be set from config or command line)
    base_seed = config['dataset'].get('base_seed', 42)
    print(f"Using base seed {base_seed} for reproducible dataset sampling")

    # Load base dataset config from file
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # --------------------------- Added for overriding dataset config from training confgig ---------------------------
    
    # Deep merge function to handle nested dictionary updates
    def deep_update(base_dict, update_dict):
        """Recursively update base_dict with values from update_dict"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    # Apply overrides from training config's dataset section
    # Exclude keys that are not meant for the dataset config itself
    exclude_keys = {'name', 'config_path', 'window_size', 'prediction_horizons', 
                   'splits', 'sample_multiplier', 'exclude_background_class', 
                   'background_class_value', 'eval_stride', 'clip_stride'}
    
    dataset_overrides = {}
    for key, value in config.get('dataset', {}).items():
        if key not in exclude_keys:
            dataset_overrides[key] = value
    
    # Apply the overrides to the base dataset config
    if dataset_overrides:
        print(f"Applying dataset config overrides: {list(dataset_overrides.keys())}")
        deep_update(dataset_config, dataset_overrides)
    
    # Print what column patterns are being used (only for debugging)
    # Hardcoded for RevalExo
    if 'modalities' in dataset_config and 'raw_imu' in dataset_config['modalities']:
        column_patterns = dataset_config['modalities']['raw_imu'].get('column_patterns', [])
        print(f"Using IMU column patterns: {column_patterns}")
        
        # Count expected channels based on patterns
        expected_channels = 0
        
        # Check for all-body patterns
        if '*acc_*' in column_patterns:
            expected_channels += 51  # 17 body parts × 3 axes
        if '*gyro_*' in column_patterns:
            expected_channels += 51  # 17 body parts × 3 axes
        
        # Check for lower body specific patterns
        lower_body_acc_patterns = ["acc_Pelvis_*", "acc_*_Upper_Leg_*", "acc_*_Lower_Leg_*", "acc_*_Foot_*"]
        lower_body_gyro_patterns = ["gyro_Pelvis_*", "gyro_*_Upper_Leg_*", "gyro_*_Lower_Leg_*", "gyro_*_Foot_*"]
        
        # If we have all lower body acc patterns but not the general *acc_*
        if all(p in column_patterns for p in lower_body_acc_patterns) and '*acc_*' not in column_patterns:
            expected_channels = 21  # 7 lower body parts × 3 axes
            
            # Check if we also have lower body gyro
            if all(p in column_patterns for p in lower_body_gyro_patterns):
                expected_channels = 42  # 7 parts × 3 axes × 2 sensors
        
        print(f"Expected number of IMU channels: {expected_channels}")
    
    # ------------------------------------------------------------------------------------------------------------

    # Get modalities
    modalities = list(config.get('modalities', {}).keys())
    
    # Get window size (use dataset default if not specified)
    window_size = config['dataset'].get('window_size', None)

    # Get prediction horizons from dataset config
    prediction_horizons = config['dataset'].get('prediction_horizons', [0])
    print(f"Extracted prediction horizons from config: {prediction_horizons}")
    
    # Get evaluation parameters
    eval_stride = config['dataset'].get('eval_stride', None)
    clip_stride = config['dataset'].get('clip_stride', None)
    
    # If not in training config, use defaults from base dataset config
    if eval_stride is None:
        eval_stride = dataset_config.get('eval_stride', 0.25)
    if clip_stride is None:
        clip_stride = dataset_config.get('clip_stride', 8.0)
    
    print(f"Evaluation parameters: eval_stride={eval_stride}s, clip_stride={clip_stride}s")

    # Get exclude background class option
    exclude_background = config['dataset'].get('exclude_background_class', False)
    background_class_value = config['dataset'].get('background_class_value', "Background")
    
    # Get video model frame size if video modality is used
    video_model_frame_size = None
    if 'video' in modalities:
        video_model_frame_size = config['modalities']['video'].get('frames_to_sample', 16)
        use_frames = config['modalities']['video'].get('use_frames', False)
    else:
        use_frames = False
    
    # Set up transforms
    train_transforms = setup_transforms(config, 'train')
    eval_transforms = setup_transforms(config, 'eval')
    
    # Get split information
    splits = config['dataset'].get('splits', {})
    train_subjects = splits.get('train_subjects', [])
    val_subjects = splits.get('val_subjects', [])
    test_subjects = splits.get('test_subjects', [])

    # Get sample multipliers from config (with defaults)
    sample_multipliers = config.get('dataset', {}).get('sample_multiplier', {})
    train_sample_multiplier = sample_multipliers.get('train', 1)
    val_sample_multiplier = sample_multipliers.get('validation', 1)
    test_sample_multiplier = sample_multipliers.get('test', 1)

    # Create datasets with the MERGED config dictionary (not path!)
    train_dataset = dataset_class(
        dataset_config=dataset_config,  # Pass the merged config dict
        modalities=modalities,
        window_size=window_size,
        video_model_frame_size=video_model_frame_size,
        subjects=train_subjects,
        split='train',
        transforms=train_transforms,
        sample_multiplier=train_sample_multiplier,
        use_frames=use_frames,
        exclude_background=exclude_background,
        background_class_value=background_class_value,
        prediction_horizons=prediction_horizons,
        eval_stride=eval_stride,
        clip_stride=clip_stride,
        base_seed=base_seed
    )
    
    val_dataset = dataset_class(
        dataset_config=dataset_config,  # Pass the merged config dict
        modalities=modalities,
        window_size=window_size,
        video_model_frame_size=video_model_frame_size,
        subjects=val_subjects,
        split='val',
        transforms=eval_transforms,
        sample_multiplier=val_sample_multiplier,
        use_frames=use_frames,
        exclude_background=exclude_background,
        background_class_value=background_class_value,
        prediction_horizons=prediction_horizons,
        eval_stride=eval_stride,
        clip_stride=clip_stride,
        base_seed=base_seed
    )
    
    test_dataset = dataset_class(
        dataset_config=dataset_config,  # Pass the merged config dict
        modalities=modalities,
        window_size=window_size,
        video_model_frame_size=video_model_frame_size,
        subjects=test_subjects,
        split='test',
        transforms=eval_transforms,
        sample_multiplier=test_sample_multiplier,
        use_frames=use_frames,
        exclude_background=exclude_background,
        background_class_value=background_class_value,
        prediction_horizons=prediction_horizons,
        eval_stride=eval_stride,
        clip_stride=clip_stride,
        base_seed=base_seed
    )
    
    # Print prediction horizons information
    if hasattr(train_dataset, 'get_prediction_horizons'):
        prediction_horizons = train_dataset.get_prediction_horizons()
        num_heads = train_dataset.get_num_prediction_heads()
        print(f"Dataset configured with {num_heads} prediction heads for horizons: {prediction_horizons}")
    
    # Get dataloader configs
    train_loader_config = config.get('dataloaders', {}).get('train', {})
    val_loader_config = config.get('dataloaders', {}).get('validation', {})
    test_loader_config = config.get('dataloaders', {}).get('test', {})
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    use_balanced_sampler = train_loader_config.get('use_balanced_sampler', False)
    if use_balanced_sampler and hasattr(train_dataset, 'get_labels'):
        # Get class distribution through labels
        try:
            print("Calculating class weights for balanced sampling...")
            labels = train_dataset.get_labels()  # This returns labels for first horizon
            unique_labels, counts = np.unique(labels, return_counts=True)
            class_counts = np.zeros(len(train_dataset.label_mapping["idx_to_label"]), dtype=np.int64)
            for label, count in zip(unique_labels, counts):
                class_counts[label] = count
            
            # Get balancing method from config
            balancing_method = train_loader_config.get('balancing_method', 'inverse')
            
            if balancing_method == 'effective':
                # Effective Number of Samples method
                beta = train_loader_config.get('beta', 0.9999)
                effective_num = 1.0 - np.power(beta, class_counts)
                weights = (1.0 - beta) / np.array(effective_num)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)  # Handle division by zero
                weights = weights / np.sum(weights) * len(weights)  # Normalize weights
                
                print(f"Using Effective Number sampling with beta={beta}")
                for i in range(len(class_counts)):
                    if class_counts[i] > 0:
                        class_name = train_dataset.label_mapping["idx_to_label"].get(str(i), f"Class {i}")
                        print(f"  {class_name}: weight={weights[i]:.6f} (count={class_counts[i]})")

            elif balancing_method == "inverse":
                # Simple inverse frequency weighting
                weights = np.zeros_like(class_counts, dtype=np.float32)
                for i, count in enumerate(class_counts):
                    if count > 0:
                        weights[i] = 1.0 / count
                    else:
                        weights[i] = 0.0
                
                print("Using simple inverse frequency weighting")
                for i in range(len(class_counts)):
                    if class_counts[i] > 0:
                        class_name = train_dataset.label_mapping["idx_to_label"].get(str(i), f"Class {i}")
                        print(f"  {class_name}: weight={weights[i]:.6f} (count={class_counts[i]})")
            
            # Create per-sample weights using label indices
            sample_weights = np.zeros(len(labels), dtype=np.float32)
            for i, label in enumerate(labels):
                sample_weights[i] = weights[label]
            
            # Convert to tensor
            sample_weights = torch.FloatTensor(sample_weights)
            shuffle = False  # sampler option is mutually exclusive with shuffle
            
            # Create appropriate sampler
            if distributed:
                train_sampler = WeightedDistributedSampler(
                    dataset=train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,  # Enable torch.multinomial weighted sampling
                    weights=sample_weights,
                    seed=base_seed
                )
            else:
                # Use a generator with fixed seed for reproducibility
                train_sampler = EpochAwareWeightedSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    base_seed=base_seed
                )
                
        except Exception as e:
            print(f"Error setting up balanced sampler: {e}. Using standard sampling instead.")
            import traceback
            traceback.print_exc()
            use_balanced_sampler = False
    else:
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=train_loader_config.get('shuffle', True),
                seed=base_seed
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=base_seed
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=base_seed
            )

    use_custom_collate = len(prediction_horizons) > 1

    # Use enhanced collate function that handles per-horizon transition flags
    if use_custom_collate:
        print("Using custom multi-horizon collate function with per-horizon transition support")
        collate_fn = multi_horizon_collate_fn
    else:
        print("Using default PyTorch collate function")
        collate_fn = None  # Use default collate function

    # Create data loaders with deterministic worker initialization
    def worker_init_fn(worker_id):
        """Initialize each worker with a unique but reproducible seed."""
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_loader_config.get('batch_size', 8),
        shuffle=(train_loader_config.get('shuffle', True) and not distributed and not use_balanced_sampler),
        num_workers=train_loader_config.get('num_workers', 4),
        pin_memory=train_loader_config.get('pin_memory', True),
        sampler=train_sampler,
        prefetch_factor=train_loader_config.get('prefetch_factor', 2) if train_loader_config.get('num_workers', 4) > 0 else None,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_loader_config.get('batch_size', 8),
        shuffle=False,
        num_workers=val_loader_config.get('num_workers', 4),
        pin_memory=val_loader_config.get('pin_memory', True),
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_loader_config.get('batch_size', 8),
        shuffle=False,
        num_workers=test_loader_config.get('num_workers', 2),
        pin_memory=test_loader_config.get('pin_memory', True),
        sampler=test_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_sampler': train_sampler
    }