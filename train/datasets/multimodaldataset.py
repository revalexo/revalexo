# datasets/multimodaldataset.py

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import re
import yaml
import glob
import torch
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import hashlib

class MultimodalSensorDataset(Dataset):
    """
    A dataset class for multimodal sensor data (IMU, video, etc.)
    Loads data from various sources based on configuration and returns windows
    of the specified modalities. Now supports future prediction at multiple horizons.
    """
    def __init__(self, 
                 dataset_config,
                 modalities=["raw_imu"],
                 window_size=None,
                 video_model_frame_size=16,
                 split="train",
                 transforms=None,
                 subjects=None,
                 sample_multiplier=1,
                 use_frames=False,
                 exclude_background=False,
                 background_class_value="Background",
                 debug_mode=False,
                 transition_window_size=0.5, # DEPRECATED - now using pre-defined transition segments from data
                 prediction_horizons=[0],
                 eval_stride=0.25,  # Stride for evaluation sampling
                 clip_stride=8.0,  # Non-overlapping portion between clips. RevalExo clips are 12s, so 8s stride means 4s overlap.
                 base_seed=42):
        """
        Initialize the dataset.
        
        Args:
            dataset_config (str or dict): Path to config file or config dictionary
            modalities (list): List of modalities to load (e.g., ["raw_imu", "video"])
            window_size (float): Size of data window in seconds, defaults to config value
            video_model_frame_size (int): Number of frames to sample from video
            split (str): Dataset split to use ('train', 'val', 'test')
            transforms (dict): Dictionary of transforms for each modality
            subjects (list): List of subject IDs to include (if None, all subjects will be included)
            sample_multiplier (int): Number of windows to extract from each sample (default: 1)
            use_frames (bool): Whether to load frames directly from frame folders instead of video (default: False)
            exclude_background (bool): Whether to exclude background samples from the dataset (default: False)
            debug_mode(bool): Whether to print the timing for data loading and transforms
            transition_window_size (float): DEPRECATED - now using pre-defined transition segments from data
            prediction_horizons (list): List of prediction horizons in seconds (e.g., [0, 0.1, 0.5, 1.0])
            eval_stride (float): Stride for deterministic evaluation sampling (default: 0.25s)
            clip_stride (float): Non-overlapping portion between clips (default: 8.0s)
            base_seed (int): Base seed for reproducible random sampling (default: 42)
        """
        # Debug mode
        self.debug_mode = debug_mode

        # Store reproducibility parameters
        self.base_seed = base_seed
        self.epoch = 0
        self.split = split
        
        # Create local random generators for this dataset instance
        self.rng = random.Random()
        self.np_rng = np.random.RandomState()
        self._update_random_state()

        # Load config if a path is provided
        if isinstance(dataset_config, str):
            with open(dataset_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dataset_config
        
        self.modalities = modalities
        # Currently supported modalities
        valid_modalities = ["raw_imu", "video", "image"] 
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Unsupported modality: {modality}. Valid options are {valid_modalities}")
        
        # Set window size (use config default if not specified)
        self.window_size = window_size if window_size is not None else self.config["default_window_size"]
        
        # Set dataset root path
        self.root_path = self.config["root_path"]

        # Store subject list and split
        self.subjects = subjects
        self.split = split
        
        # Evaluation parameters
        self.eval_stride = eval_stride
        self.clip_stride = clip_stride
        
        # Exclude background samples if specified (implemented as skipping any sample that contains the background class in its activity segments if needed)
        # For RevalExo, we don't have the background class
        self.exclude_background = exclude_background
        self.background_class_value = background_class_value

        # Prediction horizons
        self.prediction_horizons = prediction_horizons
        
        # Get clip duration from video config (if available)
        video_config = self.config.get("modalities", {}).get("video", {})
        self.clip_duration = video_config.get("clip_duration", None)
        
        # Validate prediction horizons
        self._validate_prediction_horizons()

        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        
        # Load data pairs and prepare samples
        self.samples = self._prepare_samples()

        # Filter out short samples (to prevent bug for short video)
        self._filter_short_samples(min_duration=10.0)

        # Sample multiplier
        self.sample_multiplier = max(1, sample_multiplier)

        # Use frames
        self.use_frames = use_frames

        # Video model frame size (for temporal windowing)
        self.video_model_frame_size = video_model_frame_size
        
        # Store transforms
        self.transforms = transforms if transforms else {}
        
        # Get video frame rate from config
        if "video" in self.config["modalities"]:
            self.video_fps = self.config["modalities"]["video"].get("frame_rate", 30)
        else:
            self.video_fps = 30

    def _update_random_state(self):
        """Update the random state based on base_seed and current epoch."""
        # Split-based could be overkill
        combined_seed = int(hashlib.md5(
            f"{self.base_seed}_{self.epoch}_{self.split}".encode()
        ).hexdigest()[:8], 16)

        self.rng.seed(combined_seed)
        self.np_rng.seed(combined_seed)

        if self.debug_mode:
            print(f"[{self.split}] Random state updated - Epoch: {self.epoch}, Seed: {combined_seed}")
        
    def set_epoch(self, epoch: int):
        """
        Set the current epoch for deterministic random sampling.
        This should be called at the beginning of each epoch during training.
        
        Args:
            epoch (int): Current epoch number
        """
        self.epoch = epoch
        self._update_random_state()
        
        if self.debug_mode:
            print(f"[{self.split}] Dataset epoch set to: {epoch}")
    
    def get_epoch(self) -> int:
        """Get the current epoch number."""
        return self.epoch

    def _calculate_max_eval_start(self):
        """Calculate the maximum valid starting position for evaluation sampling."""
        max_horizon = max(self.prediction_horizons)
        
        # Two constraints:
        # 1. Should not exceed clip stride (for non-overlapping evaluation)
        # 2. Should not cause window+horizon to exceed clip duration
        
        # Constraint 1: Based on clip stride
        stride_based_max = self.clip_stride - self.eval_stride
        
        # Constraint 2: Based on clip duration
        if self.clip_duration is not None:
            duration_based_max = self.clip_duration - self.window_size - max_horizon
            # Round down to nearest eval_stride multiple
            duration_based_max = (duration_based_max // self.eval_stride) * self.eval_stride
        else:
            duration_based_max = stride_based_max
        
        # Take the minimum of both constraints
        max_eval_start = min(stride_based_max, duration_based_max)
        
        # Ensure it's not negative
        max_eval_start = max(0, max_eval_start)
        
        return max_eval_start
    
    def _validate_prediction_horizons(self):
        """Validate prediction horizons against clip duration with checks for evaluation."""
        if not isinstance(self.prediction_horizons, list):
            raise ValueError("prediction_horizons must be a list")
        
        if len(self.prediction_horizons) == 0:
            raise ValueError("prediction_horizons cannot be empty")
        
        # Check for negative horizons
        for horizon in self.prediction_horizons:
            if horizon < 0:
                raise ValueError(f"Prediction horizon {horizon} cannot be negative")
        
        max_horizon = max(self.prediction_horizons)
        
        # Enhanced validation for evaluation splits
        if self.split in ['val', 'test']:
            # Check if window size itself is too large
            if self.clip_duration is not None and self.window_size > self.clip_duration:
                raise ValueError(
                    f"Window size ({self.window_size}s) exceeds clip duration ({self.clip_duration}s)!\n"
                    f"This configuration is invalid."
                )
            
            # Calculate actual maximum evaluation start position
            max_eval_start = self._calculate_max_eval_start()
            
            # Check if we can get at least one valid evaluation sample
            if max_eval_start < 0:
                raise ValueError(
                    f"Configuration invalid for deterministic evaluation:\n"
                    f"  Clip duration: {self.clip_duration}s\n"
                    f"  Window size: {self.window_size}s\n"
                    f"  Max horizon: {max_horizon}s\n"
                    f"  Cannot find any valid evaluation position!\n"
                    f"  Window + max_horizon = {self.window_size + max_horizon}s exceeds clip duration."
                )
            
            # Calculate how many samples we can actually get
            num_eval_positions = int(max_eval_start / self.eval_stride) + 1
            
            print(f"Evaluation configuration:")
            print(f"  Max valid start position: {max_eval_start}s")
            print(f"  Number of eval positions per clip: {num_eval_positions}")
            print(f"  Positions: {[i * self.eval_stride for i in range(num_eval_positions)]}")
            
            # Verify the last position is valid
            last_position = (num_eval_positions - 1) * self.eval_stride
            if self.clip_duration is not None:
                if last_position + self.window_size + max_horizon > self.clip_duration:
                    raise ValueError(
                        f"Internal validation error: Last position {last_position}s would exceed clip duration!"
                    )
        else:
            # Standard validation for training
            if self.clip_duration is not None:
                if max_horizon >= self.clip_duration:
                    raise ValueError(
                        f"Maximum prediction horizon ({max_horizon}s) must be less than "
                        f"clip duration ({self.clip_duration}s). Current horizons: {self.prediction_horizons}"
                    )
                
                if self.window_size + max_horizon > self.clip_duration:
                    raise ValueError(
                        f"Window size ({self.window_size}s) + maximum prediction horizon ({max_horizon}s) "
                        f"= {self.window_size + max_horizon}s exceeds clip duration ({self.clip_duration}s)"
                    )
        
        print(f"Using prediction horizons: {self.prediction_horizons}")
        if self.clip_duration:
            print(f"Clip duration: {self.clip_duration}s")
        if self.split in ['val', 'test']:
            print(f"Evaluation mode: deterministic sampling with stride {self.eval_stride}s")

    def get_prediction_horizons(self):
        """Get the prediction horizons for this dataset."""
        return self.prediction_horizons.copy()

    def get_num_prediction_heads(self):
        """Get the number of prediction heads needed."""
        return len(self.prediction_horizons)

    def _load_label_mapping(self):
        """Load label mapping from the specified file."""
        label_mapping_path = os.path.join(self.root_path, self.config["label_mapping_file"])
        with open(label_mapping_path, 'r') as f:
            original_mapping = json.load(f)
        
        # Unused for RevalExo
        background_label_str_to_exclude = self.background_class_value
        if not self.exclude_background:
            return original_mapping
        
        original_label_to_idx = original_mapping.get("label_to_idx", {})
        if background_label_str_to_exclude not in original_label_to_idx:
             print(f"Warning: Background label '{background_label_str_to_exclude}' not found as a key in label_to_idx mapping. No exclusion applied.")
             self.num_classes = len(original_mapping.get("idx_to_label", {}))
             return original_mapping

        adjusted_mapping = {"label_to_idx": {}, "idx_to_label": {}}
        new_idx = 0
        original_idx_to_label = original_mapping.get("idx_to_label", {})

        # Sort original items by index (key of idx_to_label, converted to int) for consistent remapping order
        try:
             sorted_original_items = sorted(
                 original_idx_to_label.items(),
                 key=lambda item: int(item[0])
             )
        except ValueError:
             print("Warning: Could not sort original idx_to_label by integer index. Using default order.")
             sorted_original_items = original_idx_to_label.items()

        # Create new mapping, skipping the entry whose LABEL matches the one to exclude
        for original_idx_str, label in sorted_original_items:
            # Compare the iterated label string with the background label string to exclude
            if label != background_label_str_to_exclude:
                adjusted_mapping["label_to_idx"][label] = new_idx
                adjusted_mapping["idx_to_label"][str(new_idx)] = label
                new_idx += 1

        # Set num_classes to the adjusted count
        self.num_classes = new_idx
        print(f"  Adjusted label mapping created. Effective number of classes: {self.num_classes}")

        return adjusted_mapping

    def _is_in_transition_window(self, timestamp, sample):
        """
        Check if a timestamp falls within any transition segment (not no_transition).
        
        Args:
            timestamp (float): Time to check
            sample (dict): Sample dict containing transition_segments
            
        Returns:
            bool: True if timestamp is in a transition window
        """
        transition_segments = sample.get('transition_segments', [])
        
        for segment in transition_segments:
            # Check if this is an actual transition (not no_transition)
            if segment.get('transition', 'no_transition') != 'no_transition':
                # Check if timestamp falls within this transition segment
                if segment['start'] <= timestamp <= segment['end']:
                    return True
        
        return False
    
    def _get_transition_flags_for_horizons(self, base_time, sample):
        """
        Get transition flags for each prediction horizon.
        
        Args:
            base_time (float): Base timestamp (end of window)
            sample (dict): Sample dict containing transition_segments
            
        Returns:
            list: List of boolean flags for each prediction horizon
        """
        transition_flags = []
        for horizon in self.prediction_horizons:
            prediction_time = base_time + horizon
            is_transition = self._is_in_transition_window(prediction_time, sample)
            transition_flags.append(is_transition)
        return transition_flags
    
    def _prepare_samples(self):
        """
        Load and prepare dataset samples from data pairs JSON files.
        Returns a list of samples with metadata, filtered by subject according to split.
        """
        samples = []
        data_pairs_folder = os.path.join(self.root_path, self.config["data_pairs_folder"])
        
        # Iterate through data pair files
        for filename in os.listdir(data_pairs_folder):
            if filename.endswith('.json'):
                data_pairs_path = os.path.join(data_pairs_folder, filename)
                
                with open(data_pairs_path, 'r') as f:
                    subject_data = json.load(f)
                
                # Extract subject from data
                subject = subject_data.get("subject", "")
                
                # Skip if subject is not in the list (if provided)
                if self.subjects is not None and subject not in self.subjects:
                    continue
                
                video_folder = os.path.join(self.root_path, subject_data["video_folder"])
                imu_folder = os.path.join(self.root_path, subject_data["imu_folder"])
                
                # Get frames folder if it exists in the JSON
                frames_folder = None
                if "frames_folder" in subject_data:
                    frames_folder = os.path.join(self.root_path, subject_data["frames_folder"])
                
                # Process each pair
                for pair_id, pair_data in subject_data["pairs"].items():    
                    # Skip if background samples are excluded
                    # Currently excludes the full clip if background class exists
                    exclude_this_pair = False
                    if self.exclude_background:
                        for segment in pair_data.get("activity_segments", []):
                            if segment.get("activity") == self.background_class_value:
                                exclude_this_pair = True
                                break

                    if exclude_this_pair:
                        continue
          
                    sample = {
                        "subject": subject,
                        "pair_id": pair_id,
                        "video_path": os.path.join(video_folder, pair_data["video"]),
                        "imu_path": os.path.join(imu_folder, pair_data["imu"]),
                        "activity_segments": pair_data["activity_segments"],
                        "transition_segments": pair_data.get("transition_segments", []),
                        "is_transition": pair_data.get("is_transition", False),
                        "has_transition_period": pair_data.get("has_transition_period", False),
                        "duration": self._get_duration(pair_data)
                    }
                    
                    # Add frames path if available
                    if "frames" in pair_data and frames_folder:
                        sample["frames_path"] = os.path.join(frames_folder, pair_data["frames"])
                    
                    samples.append(sample)
        
        if len(samples) == 0:
            if self.subjects:
                print(f"Warning: No samples found for subjects: {self.subjects}")
            else:
                print(f"Warning: No samples found in the dataset")
        
        return samples

    def _filter_short_samples(self, min_duration=10.0):
        """Remove samples shorter than min_duration seconds.""" # Might affect some data that are trimmed short towards the end
        
        original_count = len(self.samples)
        
        # Filter samples
        self.samples = [s for s in self.samples if s.get('duration', 0) >= min_duration]
        
        removed_count = original_count - len(self.samples)
        
        if removed_count > 0:
            print(f"Filtered out {removed_count} samples shorter than {min_duration}s")
            print(f"Remaining samples: {len(self.samples)}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples remaining after filtering (min duration: {min_duration}s)")
        
        return self.samples
    
    def _get_duration(self, pair_data):
        """Extract duration from activity segments."""
        max_end = 0
        for segment in pair_data["activity_segments"]:
            if segment["end"] > max_end:
                max_end = segment["end"]
        return max_end
    
    def _get_label_at_time(self, activity_segments, time):
        """Get activity label at specific timestamp."""
        for segment in activity_segments:
            if segment["start"] <= time <= segment["end"]:
                return segment["activity"]
        return "Background"  # Default if no matching segment

    def _get_labels_at_horizons(self, activity_segments, base_time):
        """Get labels for all prediction horizons from a base time."""
        labels = []
        for horizon in self.prediction_horizons:
            prediction_time = base_time + horizon
            label_str = self._get_label_at_time(activity_segments, prediction_time)
            label_idx = self.label_mapping["label_to_idx"].get(label_str, 0)
            labels.append(label_idx)
        return labels

    def get_class_distribution(self):
        """
        Calculate class distribution based on activity segments.
        Returns a dictionary or array with counts for each class.
        
        Returns:
            numpy.ndarray: Array where index corresponds to class index and value is the count
        """
        # Initialize counters for each class
        num_classes = len(self.label_mapping["idx_to_label"])
        class_counts = np.zeros(num_classes, dtype=np.int64)
        
        # Count for original samples (using first horizon for distribution calculation)
        for sample in self.samples:
            # For each sample, determine the label at potential sampling points
            max_horizon = max(self.prediction_horizons)
            max_start_time = max(0, sample["duration"] - self.window_size - max_horizon)
            
            if max_start_time <= 0:
                # If window + horizon is larger than sample, just get label at the end
                labels = self._get_labels_at_horizons(sample["activity_segments"], sample["duration"])
                label_idx = labels[0]  # Use first horizon for distribution
                class_counts[label_idx] += self.sample_multiplier
            else:
                # Divide the sampling range into equal segments based on multiplier
                segment_size = max_start_time / max(1, self.sample_multiplier)
                
                for segment_idx in range(self.sample_multiplier):
                    # Calculate segment bounds
                    segment_start = segment_size * segment_idx
                    segment_end = min(max_start_time, segment_start + segment_size)
                    
                    # Use midpoint of segment as representative
                    start_time = (segment_start + segment_end) / 2
                    end_time = min(start_time + self.window_size, sample["duration"])
                    
                    # Get labels for all horizons and use first one for distribution
                    labels = self._get_labels_at_horizons(sample["activity_segments"], end_time)
                    label_idx = labels[0]  # Use first horizon for distribution
                    class_counts[label_idx] += 1
        
        return class_counts

    def get_label_at_index(self, idx):
        """
        Get the labels for all prediction horizons at a specific index without loading the data.
        Useful for efficient sampling strategies.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            list: List of label indices for each prediction horizon
        """
        # Use the appropriate sample selection method based on split
        if self.split in ['val', 'test']:
            sample, start_time, end_time = self._get_evaluation_sample(idx)
        elif self.sample_multiplier > 1:
            sample, start_time, end_time = self._get_multiplied_sample(idx)
        else:
            # Original sample selection logic
            sample_idx = idx
            sample = self.samples[sample_idx]
            
            # Determine valid time range for window selection (accounting for max horizon)
            max_horizon = max(self.prediction_horizons)
            max_start_time = max(0, sample["duration"] - self.window_size - max_horizon)
            
            # Use a deterministic position for consistent results
            # (midpoint of valid range)
            if max_start_time > 0:
                start_time = max_start_time / 2
            else:
                start_time = 0
                
            end_time = min(start_time + self.window_size, sample["duration"])
        
        # Get labels for all prediction horizons
        return self._get_labels_at_horizons(sample["activity_segments"], end_time)

    def get_labels(self):
        """
        Get all labels for the dataset.
        This is useful for samplers that need the label information.
        
        Returns:
            list: List of labels for each sample (using first horizon for compatibility)
        """
        # Cache labels if not already done
        if not hasattr(self, "_cached_labels") or self._cached_labels is None:
            all_horizon_labels = [self.get_label_at_index(i) for i in range(len(self))]
            # Return only first horizon for compatibility with existing sampling logic
            self._cached_labels = [labels[0] for labels in all_horizon_labels]
        return self._cached_labels

    def _get_evaluation_sample(self, idx):
        """
        Get sample and window time positions for deterministic evaluation.
        
        Args:
            idx (int): The virtual index of the sample
            
        Returns:
            tuple: (sample, start_time, end_time)
        """
        # Calculate actual number of valid positions per clip
        max_eval_start = self._calculate_max_eval_start()
        samples_per_clip = int(max_eval_start / self.eval_stride) + 1
        
        # Determine which clip and position within clip
        clip_idx = idx // samples_per_clip
        position_idx = idx % samples_per_clip
        
        # Get the actual sample
        if clip_idx >= len(self.samples):
            raise IndexError(f"Clip index {clip_idx} out of range for {len(self.samples)} samples")
        
        sample = self.samples[clip_idx]
        
        # Calculate start time for this position
        start_time = position_idx * self.eval_stride
        
        # Double-check that this position is valid
        max_horizon = max(self.prediction_horizons)
        if start_time + self.window_size + max_horizon > sample["duration"]:
            # This shouldn't happen with proper validation, but add safeguard
            print(f"Warning: Evaluation sample at position {position_idx} would exceed clip duration")
            print(f"  Start time: {start_time}s")
            print(f"  Window size: {self.window_size}s")
            print(f"  Max horizon: {max_horizon}s")
            print(f"  Total reach: {start_time + self.window_size + max_horizon}s")
            print(f"  Clip duration: {sample['duration']}s")
            # Adjust start time to stay within bounds
            start_time = min(start_time, sample["duration"] - self.window_size - max_horizon)
            start_time = max(0, start_time)
        
        end_time = start_time + self.window_size
        
        return sample, start_time, end_time

    def _load_image_data(self, video_path, frames_path, start_time, end_time, use_frames, use_end_frame=True):
        """
        Load a single image (center or end frame) from video window.
        FIXED: Now directly extracts single frame instead of loading 16 frames first.

        Args:
            video_path: Path to video file
            frames_path: Path to frames folder (if available)
            start_time: Start time of window
            end_time: End time of window
            use_frames: Whether to use frames folder or video file
            use_end_frame: If True, use end frame; if False, use center frame
            
        Returns:
            Image tensor of shape [C, H, W]
        """
        try:
            # Determine target time
            if use_end_frame:
                target_time = end_time
            else:
                target_time = (start_time + end_time) / 2.0
            
            if use_frames and frames_path:
                # Load from frames folder
                fps = self.video_fps
                target_frame_idx = int(target_time * fps)
                
                # Load the single target frame
                frame_path = os.path.join(frames_path, f"frame_{target_frame_idx:04d}.jpg")
                
                if os.path.exists(frame_path):
                    img = Image.open(frame_path).convert('RGB')
                    frame = np.array(img)
                else:
                    # Find closest frame
                    closest_frame = self._find_closest_frame(frames_path, target_frame_idx)
                    if closest_frame:
                        img = Image.open(closest_frame).convert('RGB')
                        frame = np.array(img)
                    else:
                        # Return dummy frame
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Direct single frame extraction from video
                import av
                
                container = av.open(video_path)
                video_stream = container.streams.video[0]
                
                # Calculate target PTS
                target_pts = int(target_time * video_stream.time_base.denominator / video_stream.time_base.numerator)
                
                # Seek directly to target time
                container.seek(target_pts, stream=video_stream)
                
                # Extract just the single frame we need
                frame = None
                for f in container.decode(video=0):
                    frame = f.to_ndarray(format="rgb24")
                    break  # Get only one frame and stop
                
                container.close()
                
                if frame is None:
                    # Return dummy frame if extraction failed
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Convert to tensor [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(frame).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # [C, H, W]
            
            # Apply transforms if available
            if "image" in self.transforms and self.transforms["image"]:
                image_tensor = self.transforms["image"](image_tensor)
            
            return image_tensor
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading image data: {e}")
                import traceback
                traceback.print_exc()
            # Return dummy image
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def _load_imu_data(self, imu_path, start_time, end_time):
        """
        Load IMU temporal window from CSV given start and end time
        """
        try:
            # Read the CSV file
            df = pd.read_csv(imu_path)
            
            # Get relevant columns based on patterns
            column_patterns = self.config["modalities"]["raw_imu"]["column_patterns"]
            relevant_columns = []
            
            for pattern in column_patterns:
                pattern_regex = re.compile(pattern.replace("*", ".*"))
                for col in df.columns:
                    if pattern_regex.fullmatch(col):
                        relevant_columns.append(col)
            
            # If we have relevant columns, keep only those
            if relevant_columns:
                df = df[relevant_columns]
            
            # Check for NaNs before filling
            has_nans_before = df.isna().any().any()
            if has_nans_before:
                # Fill NaNs with zeros
                df = df.fillna(0)
                
            # Get sampling rate from config
            sampling_rate = self.config["modalities"]["raw_imu"]["sampling_rate"]
            
            # Calculate start and end indices
            start_idx = int(start_time * sampling_rate)
            end_idx = int(end_time * sampling_rate)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(df), end_idx)
            
            # Extract the data window
            df_window = df.iloc[start_idx:end_idx]
            
            # Convert to numpy array
            imu_data = df_window.values.astype(np.float32)
            
            return imu_data
        
        except Exception as e:
            print(f"Error loading IMU data from {imu_path}: {e}")
            sampling_rate = self.config["modalities"]["raw_imu"]["sampling_rate"]
            dummy_shape = (int(self.window_size * sampling_rate), len(relevant_columns) if relevant_columns else 1)
            # TODO: Determine how to handle loading errors
            return np.zeros(dummy_shape, dtype=np.float32)

    def load_single_frame(self, frame_path):
        """Load a single frame from full frame path."""
        if os.path.exists(frame_path):
            try:
                img = Image.open(frame_path).convert('RGB')
                return np.array(img)
            except Exception as e:
                print(f"Could not load image from {frame_path}: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Extract index from filename
            try:
                idx = int(os.path.splitext(os.path.basename(frame_path))[0].split("_")[1])
            except Exception as e:
                print(f"Could not parse frame index from {frame_path}: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            closest_frame = self._find_closest_frame(os.path.dirname(frame_path), idx)
            if closest_frame:
                try:
                    img = Image.open(closest_frame).convert('RGB')
                    return np.array(img)
                except Exception as e:
                    print(f"Could not load closest frame {closest_frame}: {e}")
                    return np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)

    def _load_video_frames(self, frames_path, start_time, end_time):
        """
        Load video data from frames folder and extract the specified time window.
        Returns tensor in [C, T, H, W] format for video transforms.
        Uses efficient temporal subsampling during loading with optimized frame loading.
        """
        # Debug mode flag
        debug = getattr(self, 'debug_mode', False)
        
        try:
            if debug:
                print("Loading frames from folder")
            
            # Get number of frames to sample
            num_frames = self.video_model_frame_size
            
            # Get frame rate from config
            fps = self.video_fps
            
            # Calculate start and end frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Count total available frames
            frame_files = glob.glob(os.path.join(frames_path, "frame_*.jpg"))
            total_frames = len(frame_files)
            
            # Ensure indices are within valid range
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            if end_frame <= start_frame:
                if debug:
                    print(f"Warning: Invalid time window for {frames_path}. Returning dummy tensor.")
                return torch.zeros((3, self.video_model_frame_size, 224, 224), dtype=torch.float32)
            
            # EFFICIENT TEMPORAL SUBSAMPLING: Calculate which frames to load
            if end_frame - start_frame <= num_frames:
                # If we have fewer frames than needed, duplicate some frames
                indices = np.array(range(start_frame, end_frame))
                if len(indices) < num_frames:
                    # Repeat frames to reach desired count
                    repeats = int(np.ceil(num_frames / len(indices)))
                    indices = np.tile(indices, repeats)[:num_frames]
            else:
                # Sample frames evenly (this is the key optimization!)
                indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
            
            # Create list of full frame paths
            frame_paths = [os.path.join(frames_path, f"frame_{idx:04d}.jpg") for idx in indices]

            # OPTIMIZED PARALLEL FRAME LOADING
            if debug:
                loading_start = time.time()
            
            with ThreadPoolExecutor(max_workers=min(4, len(frame_paths))) as executor:
                frames = list(executor.map(self.load_single_frame, frame_paths))
            
            if debug:
                loading_end = time.time()
                print(f"Parallel frame loading: {(loading_end - loading_start)*1000:.2f}ms")
            
            if not frames:
                return torch.zeros((3, self.video_model_frame_size, 224, 224), dtype=torch.float32)
            
            # Stack frames into array [T, H, W, C]
            if debug:
                stack_start = time.time()
            
            frames = np.stack(frames)
            
            if debug:
                stack_end = time.time()
                print(f"Stack frames: {(stack_end - stack_start)*1000:.2f}ms")
            
            # Convert to tensor and change to [C, T, H, W] format
            if debug:
                tensor_start = time.time()
            
            video_tensor = torch.from_numpy(frames).float()  # [T, H, W, C]
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
            
            if debug:
                tensor_end = time.time()
                print(f"Tensor conversion: {(tensor_end - tensor_start)*1000:.2f}ms, tensor shape: {video_tensor.shape}")
            
            # Apply transforms
            if "video" in self.transforms and self.transforms["video"]:
                if debug and hasattr(self.transforms["video"], 'transforms'):
                    # Debug mode: time each transform
                    current_tensor = video_tensor
                    for i, transform in enumerate(self.transforms["video"].transforms):
                        transform_start = time.time()
                        current_tensor = transform(current_tensor)
                        transform_end = time.time()
                        print(f"Transform {i} ({transform.__class__.__name__}): {(transform_end - transform_start)*1000:.2f}ms, output shape: {current_tensor.shape}")
                    video_tensor = current_tensor
                else:
                    # Normal mode: just apply transforms
                    video_tensor = self.transforms["video"](video_tensor)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video frames from {frames_path}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            # TODO: Determine how to handle loading errors
            return torch.zeros((3, self.video_model_frame_size, 224, 224), dtype=torch.float32)

    def _find_closest_frame(self, frames_path, target_idx):
        """Find the closest frame to the target index in the frames folder (Optimized)"""
        try:
            # Use glob pattern for better performance
            frame_pattern = os.path.join(frames_path, "frame_*.jpg")
            frame_files = glob.glob(frame_pattern)
            
            if not frame_files:
                return None
            
            # Extract frame indices from filenames more efficiently
            frame_indices = []
            for frame_file in frame_files:
                # Extract the frame number from the filename
                base_name = os.path.basename(frame_file)
                match = re.search(r"frame_(\d+)\.jpg", base_name)
                if match:
                    frame_idx = int(match.group(1))
                    frame_indices.append((frame_idx, frame_file))
            
            if not frame_indices:
                return None
            
            # Find the closest frame by index
            closest_frame = min(frame_indices, key=lambda x: abs(x[0] - target_idx))
            return closest_frame[1]
            
        except Exception as e:
            print(f"Error finding closest frame: {e}")
            return None

    def _load_video_data(self, video_path, start_time, end_time):
        """
        Load video data (finally using PyAV for efficient (most effective so far) decoding).
        Returns tensor in [C, T, H, W] format for video transforms.
        Uses efficient temporal subsampling during loading.
        """
        # Debug mode flag
        # To determine the bottleneck step during video loading
        debug = getattr(self, 'debug_mode', False)
        
        try:
            import av
            
            # Number of frames to extract
            num_frames = self.video_model_frame_size
            
            # Open the video container
            if debug:
                av_start = time.time()
            
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            # Calculate exact timestamps to seek to
            start_pts = int(start_time * video_stream.time_base.denominator / video_stream.time_base.numerator)
            end_pts = int(end_time * video_stream.time_base.denominator / video_stream.time_base.numerator)
            
            # Seek to the start timestamp
            container.seek(start_pts, stream=video_stream)
            
            if debug:
                av_end = time.time()
                print(f"Video open/seek: {(av_end - av_start)*1000:.2f}ms")
            
            # Decode all frames within the range
            if debug:
                decode_start = time.time()
            
            all_frames = []
            for frame in container.decode(video=0):
                if frame.pts < start_pts:
                    continue
                if frame.pts > end_pts:
                    break
                all_frames.append(frame.to_ndarray(format="rgb24"))
            
            container.close()
            
            if debug:
                decode_end = time.time()
                print(f"Video decode: {(decode_end - decode_start)*1000:.2f}ms, decoded {len(all_frames)} frames")
            
            if len(all_frames) == 0:
                if debug:
                    print(f"Warning: No frames decoded in range {start_time}-{end_time} for {video_path}")
                return torch.zeros((3, self.video_model_frame_size, 224, 224), dtype=torch.float32)
            
            # EFFICIENT TEMPORAL SUBSAMPLING: Select only the frames we need
            if debug:
                subsample_start = time.time()
            
            if len(all_frames) < num_frames:
                # If we have fewer frames than needed, duplicate frames
                indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
                selected_frames = [all_frames[i] for i in indices]
            else:
                # Sample frames evenly
                indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
                selected_frames = [all_frames[i] for i in indices]
            
            if debug:
                subsample_end = time.time()
                print(f"Temporal subsampling: {(subsample_end - subsample_start)*1000:.2f}ms, selected {len(selected_frames)} frames")
            
            # Stack frames into array [T, H, W, C]
            if debug:
                stack_start = time.time()
            
            frames = np.stack(selected_frames)
            
            if debug:
                stack_end = time.time()
                print(f"Stack frames: {(stack_end - stack_start)*1000:.2f}ms")
            
            # Convert to tensor and change to [C, T, H, W] format
            if debug:
                tensor_start = time.time()
            
            video_tensor = torch.from_numpy(frames).float()  # [T, H, W, C]
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
            
            if debug:
                tensor_end = time.time()
                print(f"Tensor conversion: {(tensor_end - tensor_start)*1000:.2f}ms, tensor shape: {video_tensor.shape}")
            
            # Apply transforms
            if "video" in self.transforms and self.transforms["video"]:
                if debug and hasattr(self.transforms["video"], 'transforms'):
                    # Debug mode: time each transform
                    current_tensor = video_tensor
                    for i, transform in enumerate(self.transforms["video"].transforms):
                        transform_start = time.time()
                        current_tensor = transform(current_tensor)
                        transform_end = time.time()
                        print(f"Transform {i} ({transform.__class__.__name__}): {(transform_end - transform_start)*1000:.2f}ms, output shape: {current_tensor.shape}")
                    video_tensor = current_tensor
                else:
                    # Normal mode: just apply transforms
                    video_tensor = self.transforms["video"](video_tensor)
            
            return video_tensor

        except Exception as e:
            print(f"Error loading video data from {video_path}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            return torch.zeros((3, self.video_model_frame_size, 224, 224), dtype=torch.float32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.split in ['val', 'test']:
            # Deterministic evaluation: calculate actual number of valid positions
            max_eval_start = self._calculate_max_eval_start()
            samples_per_clip = int(max_eval_start / self.eval_stride) + 1
            return len(self.samples) * samples_per_clip
        else:
            # Training: use sample multiplier
            return len(self.samples) * self.sample_multiplier
    
    def _get_multiplied_sample(self, idx):
        """
        Used to multiply samples from a clip (setting this to for example 3 means that we get 3 samples from one 12-second clip within one training epoch)
        Get sample and window time positions for the multiplied dataset.
        Now accounts for prediction horizons in time bounds.
        
        Args:
            idx (int): The virtual index of the sample
            
        Returns:
            tuple: (sample, start_time, end_time)
        """
        # Calculate the actual sample index and segment index
        sample_idx = idx % len(self.samples)
        segment_idx = idx // len(self.samples)
        
        # Get sample metadata
        sample = self.samples[sample_idx]
        
        # Determine valid time range for window selection (accounting for max horizon)
        max_horizon = max(self.prediction_horizons)
        max_start_time = max(0, sample["duration"] - self.window_size - max_horizon)
        
        if max_start_time <= 0:
            # No valid range, just start at 0
            start_time = 0
        else:
            # Divide the valid range into segments based on the multiplier
            segment_size = max_start_time / self.sample_multiplier
            
            # Calculate the start and end of this segment
            segment_start = segment_size * segment_idx
            segment_end = min(max_start_time, segment_start + segment_size)

            # Use local random generator for reproducible randomness
            # Create a unique seed for this specific sample
            sample_seed = self.base_seed + self.epoch * 1000 + idx
            local_rng = random.Random(sample_seed)
            
            # Select a random point within this segment
            start_time = local_rng.uniform(segment_start, segment_end)
        
        end_time = min(start_time + self.window_size, sample["duration"])
        
        return sample, start_time, end_time

    def __getitem__(self, idx):
        """
        Get a data sample for a given index.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: Data for each requested modality, plus label indices for all horizons,
                   plus transition flags list, plus subject ID string
        """
        # Use the appropriate sample selection method based on split
        if self.split in ['val', 'test']:
            # Deterministic evaluation sampling
            sample, start_time, end_time = self._get_evaluation_sample(idx)
        elif self.sample_multiplier > 1:
            sample, start_time, end_time = self._get_multiplied_sample(idx)
        else:
            # Original sample selection logic for training
            sample_idx = idx
            sample = self.samples[sample_idx]
            
            # Determine valid time range for window selection (accounting for max horizon)
            max_horizon = max(self.prediction_horizons)

            # SAFE BUFFER: Leave 0.5s buffer from the end to avoid edge cases
            safe_duration = sample["duration"] - 0.5
            max_start_time = max(0, safe_duration - self.window_size - max_horizon)
            
            # Use reproducible random selection within valid range
            if max_start_time > 0:
                # Create a unique seed for this specific sample in this epoch
                sample_seed = self.base_seed + self.epoch * 1000 + idx
                local_rng = random.Random(sample_seed)
                start_time = local_rng.uniform(0, max_start_time)
            else:
                start_time = 0
                
            end_time = min(start_time + self.window_size, sample["duration"])
        
        # Get labels for all prediction horizons
        horizon_labels = self._get_labels_at_horizons(sample["activity_segments"], end_time)
        
        # Get per-horizon transition flags
        transition_flags = self._get_transition_flags_for_horizons(end_time, sample)
        
        # Load requested modalities
        result = {}
            
        if "raw_imu" in self.modalities:
            imu_data = self._load_imu_data(sample["imu_path"], start_time, end_time)
            
            # Defensive checks for IMU data
            if imu_data is not None:
                expected_imu_length = int(self.window_size * self.config["modalities"]["raw_imu"]["sampling_rate"])
                
                # Ensure correct shape
                if imu_data.shape[0] != expected_imu_length:
                    if self.debug_mode:
                        print(f"Warning: IMU data length {imu_data.shape[0]} doesn't match expected {expected_imu_length}")
                    
                    # Pad or truncate as needed
                    if imu_data.shape[0] < expected_imu_length:
                        # Pad with zeros
                        padding = expected_imu_length - imu_data.shape[0]
                        imu_data = np.pad(imu_data, ((0, padding), (0, 0)), mode='constant')
                    elif imu_data.shape[0] > expected_imu_length:
                        # Truncate
                        imu_data = imu_data[:expected_imu_length]
                
                # Apply transforms if available
                # Data dimensions are managed through the transforms (for example, through transpose based on what the model expects)
                if "raw_imu" in self.transforms and self.transforms["raw_imu"]:
                    imu_data = self.transforms["raw_imu"](imu_data)

                # Ensure the array is contiguous
                imu_data = np.ascontiguousarray(imu_data)
                result["raw_imu"] = imu_data
        
        if "video" in self.modalities:
            if self.use_frames and "frames_path" in sample:
                # Load video from frames if use_frames is True and frames_path is available
                result["video"] = self._load_video_frames(sample["frames_path"], start_time, end_time)
            else:
                # Load video from file
                result["video"] = self._load_video_data(sample["video_path"], start_time, end_time)
        
        if "image" in self.modalities:
            # Load single image (center frame)
            use_frames = self.use_frames if hasattr(self, 'use_frames') else False
            frames_path = sample.get("frames_path", None)
            
            image_data = self._load_image_data(
                sample["video_path"], 
                frames_path,
                start_time, 
                end_time,
                use_frames
            )
            result["image"] = image_data

        # Prepare return values
        return_values = []
        for modality in self.modalities:
            return_values.append(result[modality])
        
        if len(self.prediction_horizons) == 1:
            # Single horizon: return single label value (not in list)
            return_values.append(horizon_labels[0]) 
        else:
            # Multi-horizon: return list of labels
            return_values.append(horizon_labels)
        
        # Return list of transition flags (one per horizon)
        return_values.append(transition_flags)

        # Return subject ID for per-subject evaluation
        return_values.append(sample["subject"])

        return tuple(return_values) if len(return_values) > 1 else return_values[0]