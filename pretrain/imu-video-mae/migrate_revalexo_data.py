#!/usr/bin/env python3
"""
Script to convert RevalExo dataset to EVI-MAE format

RevalExo has 7 IMU sensors (lower body only):
- Pelvis
- Right_Upper_Leg, Right_Lower_Leg, Right_Foot
- Left_Upper_Leg, Left_Lower_Leg, Left_Foot

To match the EVI-MAE model architecture (which expects 4 body parts x 3 axes = 12 channels),
we select 4 sensors from RevalExo:
- left_arm  -> Left_Foot (distal left)
- right_arm -> Right_Foot (distal right)
- left_leg  -> Left_Upper_Leg (proximal left)
- right_leg -> Right_Upper_Leg (proximal right)
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple

class RevalExoToEVIMAE:
    def __init__(self,
                 source_dir: str = "/path/to/RevalExoDataset",
                 target_dir: str = "./data_release/revalexo-release",
                 train_ratio: float = 0.8,
                 random_seed: int = 42):

        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        random.seed(random_seed)
        np.random.seed(random_seed)

        # IMU channel mapping for RevalExo
        # Original column indices in the CSV file (0-indexed):
        # acc_Pelvis: 0,1,2
        # acc_Right_Upper_Leg: 3,4,5
        # acc_Right_Lower_Leg: 6,7,8
        # acc_Right_Foot: 9,10,11
        # acc_Left_Upper_Leg: 12,13,14
        # acc_Left_Lower_Leg: 15,16,17
        # acc_Left_Foot: 18,19,20
        #
        # EVI-MAE expects 4 body parts in order: left_arm, right_arm, left_leg, right_leg
        # We map RevalExo sensors to match this structure:
        self.channel_mapping = {
            'left_arm': [18, 19, 20],    # Left_Foot (distal left)
            'right_arm': [9, 10, 11],    # Right_Foot (distal right)
            'left_leg': [12, 13, 14],    # Left_Upper_Leg (proximal left)
            'right_leg': [3, 4, 5]       # Right_Upper_Leg (proximal right)
        }

        # Order of body parts (must match EVI-MAE expected order)
        self.sensor_order = ['left_arm', 'right_arm', 'left_leg', 'right_leg']

    def setup_directories(self):
        """Create the directory structure matching WEAR dataset"""
        print("Setting up directory structure...")

        # Create main directories
        (self.target_dir / "cav_label").mkdir(parents=True, exist_ok=True)
        (self.target_dir / "cav_label" / "train_pretrain").mkdir(exist_ok=True)
        (self.target_dir / "cav_label" / "test_pretrain").mkdir(exist_ok=True)
        (self.target_dir / "cav_label" / "train_finetune").mkdir(exist_ok=True)
        (self.target_dir / "cav_label" / "test_finetune").mkdir(exist_ok=True)

        (self.target_dir / "trim_12s_IMU").mkdir(exist_ok=True)
        (self.target_dir / "trim_12s_videos").mkdir(exist_ok=True)

        # Create class labels CSV (for now, using label 0 for pretraining)
        class_labels = pd.DataFrame({
            'index': [0],
            'mid': ['pretrain'],
            'display_name': ['Pretraining']
        })
        class_labels.to_csv(self.target_dir / "cav_label" / "class_labels_indices.csv", index=False)

    def convert_imu_data(self, source_imu_path: Path, target_imu_path: Path):
        """Convert IMU data to EVI-MAE format (21 channels for RevalExo)"""
        # Read the original IMU data with headers
        df = pd.read_csv(source_imu_path)

        # Extract only the required channels in the correct order
        selected_columns = []
        for sensor in self.sensor_order:
            for idx in self.channel_mapping[sensor]:
                selected_columns.append(df.columns[idx])

        # Create new dataframe with selected columns
        new_data = df[selected_columns].values

        # Save without header (matching WEAR format)
        np.savetxt(target_imu_path, new_data, delimiter=',', fmt='%.6f')
        return new_data

    def copy_video(self, source_video_path: Path, target_video_path: Path):
        """Copy video file to target location"""
        shutil.copy2(source_video_path, target_video_path)

    def process_data_pairs(self) -> Tuple[List[Dict], List[Dict], List[np.ndarray]]:
        """Process data pairs and create train/test splits"""
        print("Processing data pairs...")

        # Read all data pair JSON files
        all_pairs = []
        all_imu_data = []  # For statistics calculation
        data_pairs_dir = self.source_dir / "data_pairs"

        skipped_no_video = 0
        skipped_missing_files = 0

        for json_file in sorted(data_pairs_dir.glob("*.json")):
            with open(json_file, 'r') as f:
                subject_data = json.load(f)
                subject = subject_data['subject']
                has_ego_video = subject_data.get('has_ego_video', False)

                # Skip subjects without ego video (can't do video-IMU pretraining)
                if not has_ego_video:
                    print(f"Skipping {subject}: no ego video available")
                    continue

                print(f"Processing {subject}...")

                for pair_id, pair_info in subject_data['pairs'].items():
                    # Skip pairs without ego video
                    if not pair_info.get('has_ego_video', has_ego_video):
                        skipped_no_video += 1
                        continue

                    # Create pair entry matching WEAR format
                    pair_entry = {
                        'video_id': pair_id,
                        'imu': f"trim_12s_IMU/{pair_id}.csv",
                        'frame_path': f"trim_12s_videos/{pair_id}.mp4",
                        'label': 0  # Using 0 for pretraining
                    }

                    # Process IMU file
                    source_imu = self.source_dir / "trim_imus" / subject / pair_info['imu']
                    target_imu = self.target_dir / "trim_12s_IMU" / f"{pair_id}.csv"

                    if not source_imu.exists():
                        print(f"Warning: IMU file not found: {source_imu}")
                        skipped_missing_files += 1
                        continue

                    # Process video file
                    source_video = self.source_dir / "trim_videos" / subject / pair_info['video']
                    target_video = self.target_dir / "trim_12s_videos" / f"{pair_id}.mp4"

                    if not source_video.exists():
                        print(f"Warning: Video file not found: {source_video}")
                        skipped_missing_files += 1
                        continue

                    # Convert IMU data
                    imu_data = self.convert_imu_data(source_imu, target_imu)
                    all_imu_data.append(imu_data)

                    # Copy video
                    self.copy_video(source_video, target_video)

                    all_pairs.append(pair_entry)

        print(f"\nSkipped {skipped_no_video} pairs without ego video")
        print(f"Skipped {skipped_missing_files} pairs with missing files")

        # Shuffle and split into train/test
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * self.train_ratio)

        train_pairs = all_pairs[:split_idx]
        test_pairs = all_pairs[split_idx:]

        print(f"\nTotal pairs: {len(all_pairs)}")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")

        return train_pairs, test_pairs, all_imu_data

    def calculate_statistics(self, all_imu_data: List[np.ndarray]) -> Tuple[float, float]:
        """Calculate mean and std from all IMU data"""
        print("\nCalculating IMU statistics...")

        if not all_imu_data:
            print("Warning: No IMU data to calculate statistics")
            return 0.0, 1.0  # Default values

        # Concatenate all IMU data
        all_data = np.concatenate(all_imu_data, axis=0)

        # Calculate statistics
        mean = np.mean(all_data)
        std = np.std(all_data)

        print(f"IMU Dataset Mean: {mean:.2f}")
        print(f"IMU Dataset Std: {std:.2f}")

        # Save statistics to file for reference
        stats_file = self.target_dir / "imu_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write(f"IMU Dataset Statistics (RevalExo)\n")
            f.write(f"Mean: {mean:.6f}\n")
            f.write(f"Std: {std:.6f}\n")
            f.write(f"Total samples: {len(all_data)}\n")
            f.write(f"Shape per sample: {all_imu_data[0].shape if all_imu_data else 'N/A'}\n")
            f.write(f"Number of channels: 12 (4 body parts x 3 axes)\n")
            f.write(f"Body part mapping:\n")
            f.write(f"  left_arm  -> Left_Foot\n")
            f.write(f"  right_arm -> Right_Foot\n")
            f.write(f"  left_leg  -> Left_Upper_Leg\n")
            f.write(f"  right_leg -> Right_Upper_Leg\n")

        return mean, std

    def save_json_splits(self, train_pairs: List[Dict], test_pairs: List[Dict]):
        """Save train and test splits as JSON files"""
        print("\nSaving JSON split files...")

        # Save pretrain splits
        train_pretrain = {'data': train_pairs}
        test_pretrain = {'data': test_pairs}

        train_file = self.target_dir / "cav_label" / "train_pretrain" / f"labels_{len(train_pairs)}.json"
        test_file = self.target_dir / "cav_label" / "test_pretrain" / f"labels_{len(test_pairs)}.json"

        with open(train_file, 'w') as f:
            json.dump(train_pretrain, f, indent=1)

        with open(test_file, 'w') as f:
            json.dump(test_pretrain, f, indent=1)

        # Also save finetune splits (same as pretrain for now)
        train_finetune_file = self.target_dir / "cav_label" / "train_finetune" / f"labels_{len(train_pairs)}.json"
        test_finetune_file = self.target_dir / "cav_label" / "test_finetune" / f"labels_{len(test_pairs)}.json"

        with open(train_finetune_file, 'w') as f:
            json.dump(train_pretrain, f, indent=1)

        with open(test_finetune_file, 'w') as f:
            json.dump(test_pretrain, f, indent=1)

        print(f"Saved train split: {train_file}")
        print(f"Saved test split: {test_file}")

    def create_setup_script(self, mean: float, std: float, train_count: int, test_count: int):
        """Create a setup script with the calculated values"""
        script_path = self.target_dir / "setup_info.sh"

        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated setup information for RevalExo dataset\n\n")
            f.write(f"# IMU Statistics\n")
            f.write(f"export IMU_DATASET_MEAN={mean:.2f}\n")
            f.write(f"export IMU_DATASET_STD={std:.2f}\n\n")
            f.write(f"# Dataset splits\n")
            f.write(f"export TRAIN_SAMPLES={train_count}\n")
            f.write(f"export TEST_SAMPLES={test_count}\n\n")
            f.write(f"# File paths\n")
            f.write(f"export TRAIN_JSON=labels_{train_count}.json\n")
            f.write(f"export TEST_JSON=labels_{test_count}.json\n\n")
            f.write(f"# IMU Configuration\n")
            f.write(f"export IMU_CHANNEL_NUM=12  # 4 body parts x 3 axes\n")

        print(f"\nSetup script created: {script_path}")
        print("Source this file before running pretrain_revalexo.sh:")
        print(f"  source {script_path}")

    def run(self):
        """Main conversion pipeline"""
        print(f"Converting RevalExo dataset from {self.source_dir}")
        print(f"Target directory: {self.target_dir}")

        # Setup directories
        self.setup_directories()

        # Process data and create splits
        train_pairs, test_pairs, all_imu_data = self.process_data_pairs()

        # Calculate statistics
        mean, std = self.calculate_statistics(all_imu_data)

        # Save JSON files
        self.save_json_splits(train_pairs, test_pairs)

        # Create setup script
        self.create_setup_script(mean, std, len(train_pairs), len(test_pairs))

        print("\n" + "="*50)
        print("Conversion complete!")
        print("="*50)
        print(f"Data saved to: {self.target_dir}")

        # Print summary statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {len(train_pairs) + len(test_pairs)}")
        print(f"Train samples: {len(train_pairs)} ({len(train_pairs)/(len(train_pairs)+len(test_pairs))*100:.1f}%)")
        print(f"Test samples: {len(test_pairs)} ({len(test_pairs)/(len(train_pairs)+len(test_pairs))*100:.1f}%)")
        print(f"IMU sampling rate: ~60 Hz (check actual data)")
        print(f"Video frame rate: ~30 fps (check actual data)")
        print(f"Sample duration: ~12 seconds")
        print(f"IMU channels: 12 (4 body parts x 3 axes)")
        print(f"Body part mapping:")
        print(f"  left_arm  -> Left_Foot")
        print(f"  right_arm -> Right_Foot")
        print(f"  left_leg  -> Left_Upper_Leg")
        print(f"  right_leg -> Right_Upper_Leg")
        print(f"IMU mean: {mean:.2f}")
        print(f"IMU std: {std:.2f}")

        print("\n=== Next Steps ===")
        print("1. Update pretrain_revalexo.sh with these values:")
        print(f"   imu_dataset_mean={mean:.2f}")
        print(f"   imu_dataset_std={std:.2f}")
        print(f"   imu_channel_num=12")
        print("2. Run the pretraining script:")
        print("   cd /home/diwas/PhD/External/IMU-Video-MAE/egs/release")
        print("   bash pretrain_revalexo.sh")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert RevalExo dataset to EVI-MAE format")
    parser.add_argument("--source_dir", type=str, default="/path/to/RevalExoDataset",
                        help="Path to the raw RevalExo dataset")
    parser.add_argument("--target_dir", type=str, default="./data_release/revalexo-release",
                        help="Output directory for converted data")
    args = parser.parse_args()
    converter = RevalExoToEVIMAE(source_dir=args.source_dir, target_dir=args.target_dir)
    converter.run()
