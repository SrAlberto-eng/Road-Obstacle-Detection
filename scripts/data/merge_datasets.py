#!/usr/bin/env python3
"""
Merge Road Obstacle Datasets - Safety-First Version
Merges 4 YOLO datasets with comprehensive validation and class remapping.

Datasets:
- animals_yolo: dog-camel-deer, Animals -> animal
- pothole_yolo: pothole -> pothole  
- road_debris_yolo: object (x2) -> debris
- self_driving_yolo: 14 vehicle/person classes -> remapped to unified classes

Final classes (6 total):
  0: vehicle     (car, truck, bus, train, trailer, other_vehicle, cars)
  1: person      (pedestrian, rider, other_person)
  2: bicycle     (bicycle, motorcycle)
  3: animal      (dog-camel-deer, Animals)
  4: pothole     (pothole)
  5: debris      (object from road_debris)

Usage:
    # Dry run first (recommended)
    python merge_road_datasets.py --output data/road_obstacles_final --dry-run
    
    # Actual merge
    python merge_road_datasets.py --output data/road_obstacles_final
"""

import argparse
import shutil
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional


class RoadDatasetMerger:
    """Merge road obstacle datasets with validation and safety checks."""
    
    def __init__(self, output_dir: str, dry_run: bool = False):
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        
        # Define datasets and their mappings
        self.datasets = {
            'animals_yolo': {
                'path': Path('data/animals_yolo'),
                'mapping': {
                    0: 3,  # dog-camel-deer -> animal
                    1: 3   # Animals -> animal
                },
                'original_classes': {0: 'dog-camel-deer', 1: 'Animals'}
            },
            'pothole_yolo': {
                'path': Path('data/pothole_yolo'),
                'mapping': {
                    0: 4   # pothole -> pothole
                },
                'original_classes': {0: 'pothole'}
            },
            'road_debris_yolo': {
                'path': Path('data/road_debris_yolo'),
                'mapping': {
                    0: 5,  # object -> debris
                    1: 5   # object -> debris (both map to debris)
                },
                'original_classes': {0: 'object', 1: 'object'}
            },
            'self_driving_yolo': {
                'path': Path('data/self_driving_yolo'),
                'mapping': {
                    0: 0,   # cars -> vehicle
                    1: 2,   # bicycle -> bicycle
                    2: 0,   # bus -> vehicle
                    3: 0,   # car -> vehicle
                    4: 2,   # motorcycle -> bicycle
                    5: 1,   # other person -> person
                    6: 0,   # other vehicle -> vehicle
                    7: 1,   # pedestrian -> person
                    8: 1,   # rider -> person
                    9: -1,  # traffic light -> SKIP (not obstacle)
                    10: -1, # traffic sign -> SKIP (not obstacle)
                    11: 0,  # trailer -> vehicle
                    12: 0,  # train -> vehicle
                    13: 0   # truck -> vehicle
                },
                'original_classes': {
                    0: 'cars', 1: 'bicycle', 2: 'bus', 3: 'car',
                    4: 'motorcycle', 5: 'other person', 6: 'other vehicle',
                    7: 'pedestrian', 8: 'rider', 9: 'traffic light',
                    10: 'traffic sign', 11: 'trailer', 12: 'train', 13: 'truck'
                }
            }
        }
        
        # Final unified classes (6 classes)
        self.final_classes = {
            0: 'vehicle',    # All vehicles
            1: 'person',     # All persons
            2: 'bicycle',    # Bicycles and motorcycles
            3: 'animal',     # Animals
            4: 'pothole',    # Road damage
            5: 'debris'      # Road debris
        }
        
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'skipped_annotations': 0,
            'class_counts': Counter(),
            'dataset_counts': Counter(),
            'split_counts': Counter()
        }
        
        self.validation_errors = []
        self.validation_warnings = []
    
    def log_error(self, message: str):
        """Log an error message."""
        self.validation_errors.append(message)
        print(f"  ❌ ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.validation_warnings.append(message)
        print(f"  ⚠️  WARNING: {message}")
    
    def validate_datasets(self) -> bool:
        """Validate all source datasets before merging."""
        print(f"\n{'='*70}")
        print(f"VALIDATION: Checking Source Datasets")
        print(f"{'='*70}")
        
        all_valid = True
        
        for dataset_name, dataset_info in self.datasets.items():
            dataset_path = dataset_info['path']
            
            print(f"\nChecking: {dataset_name}")
            
            # Check if dataset exists
            if not dataset_path.exists():
                self.log_error(f"{dataset_name} not found at {dataset_path}")
                all_valid = False
                continue
            
            # Check data.yaml
            yaml_path = dataset_path / 'data.yaml'
            if not yaml_path.exists():
                self.log_error(f"{dataset_name}: data.yaml not found")
                all_valid = False
                continue
            
            # Load and validate yaml
            try:
                with open(yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                
                if 'names' not in data_yaml:
                    self.log_error(f"{dataset_name}: 'names' not found in data.yaml")
                    all_valid = False
                    continue
                
                # Verify class count
                num_classes = len(data_yaml['names'])
                expected_classes = len(dataset_info['original_classes'])
                
                if num_classes != expected_classes:
                    self.log_warning(
                        f"{dataset_name}: Expected {expected_classes} classes, found {num_classes}"
                    )
                
                print(f"  ✓ Found {num_classes} classes")
                
            except Exception as e:
                self.log_error(f"{dataset_name}: Failed to load data.yaml - {e}")
                all_valid = False
                continue
            
            # Check for at least one split
            found_split = False
            for split in ['train', 'val', 'test']:
                images_dir = dataset_path / 'images' / split
                labels_dir = dataset_path / 'labels' / split
                
                if images_dir.exists() and labels_dir.exists():
                    image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
                    label_count = len(list(labels_dir.glob('*.txt')))
                    
                    if image_count > 0:
                        print(f"  ✓ {split}: {image_count} images, {label_count} labels")
                        found_split = True
            
            if not found_split:
                self.log_error(f"{dataset_name}: No valid splits found")
                all_valid = False
        
        print(f"\n{'='*70}")
        if all_valid:
            print(f"✓ ALL DATASETS VALID")
        else:
            print(f"❌ VALIDATION FAILED")
        print(f"{'='*70}")
        
        return all_valid
    
    def remap_label_file(self, label_path: Path, class_mapping: dict) -> List[str]:
        """Remap class IDs in a label file with validation."""
        if not label_path.exists():
            return []
        
        remapped_lines = []
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        if len(parts) > 0:
                            self.log_warning(
                                f"Invalid annotation in {label_path.name} line {line_num}: "
                                f"Expected 5+ values, got {len(parts)}"
                            )
                        continue
                    
                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        self.log_warning(
                            f"Invalid class ID in {label_path.name} line {line_num}: {parts[0]}"
                        )
                        continue
                    
                    bbox = parts[1:5]
                    
                    # Validate bbox coordinates
                    try:
                        x_center, y_center, width, height = map(float, bbox)
                        
                        # Check if coordinates are normalized (0-1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 < width <= 1 and 0 < height <= 1):
                            self.log_warning(
                                f"Out of bounds bbox in {label_path.name} line {line_num}"
                            )
                            continue
                        
                    except ValueError:
                        self.log_warning(
                            f"Invalid bbox values in {label_path.name} line {line_num}"
                        )
                        continue
                    
                    # Remap class ID
                    if old_class_id not in class_mapping:
                        self.log_warning(
                            f"Unknown class {old_class_id} in {label_path.name}, skipping"
                        )
                        continue
                    
                    new_class_id = class_mapping[old_class_id]
                    
                    # Skip if mapped to -1 (traffic signs/lights)
                    if new_class_id == -1:
                        self.stats['skipped_annotations'] += 1
                        continue
                    
                    remapped_lines.append(f"{new_class_id} {' '.join(bbox)}")
                    self.stats['class_counts'][self.final_classes[new_class_id]] += 1
                    
        except Exception as e:
            self.log_error(f"Failed to process {label_path.name}: {e}")
            return []
        
        return remapped_lines
    
    def merge_split(self, split_name: str):
        """Merge one split (train/val/test) from all datasets."""
        print(f"\n{'='*70}")
        print(f"Merging Split: {split_name.upper()}")
        if self.dry_run:
            print(f"[DRY RUN MODE - No files will be modified]")
        print(f"{'='*70}")
        
        if not self.dry_run:
            # Create output directories
            output_images = self.output_dir / 'images' / split_name
            output_labels = self.output_dir / 'labels' / split_name
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)
        
        split_stats = {
            'images': 0,
            'annotations': 0,
            'skipped': 0
        }
        
        # Process each dataset
        for dataset_name, dataset_info in self.datasets.items():
            dataset_path = dataset_info['path']
            class_mapping = dataset_info['mapping']
            
            print(f"\nProcessing: {dataset_name}")
            
            # Check if split exists
            images_dir = dataset_path / 'images' / split_name
            labels_dir = dataset_path / 'labels' / split_name
            
            if not images_dir.exists():
                print(f"  Split '{split_name}' not found, skipping")
                continue
            
            # Get all images
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            if not image_files:
                print(f"  No images found, skipping")
                continue
            
            print(f"  Found {len(image_files)} images")
            
            dataset_annotations = 0
            dataset_skipped = 0
            
            for img_file in image_files:
                # Create unique filename (prefix with dataset name)
                new_name = f"{dataset_name}_{img_file.name}"
                
                if not self.dry_run:
                    dest_image = self.output_dir / 'images' / split_name / new_name
                    dest_label = self.output_dir / 'labels' / split_name / f"{Path(new_name).stem}.txt"
                    
                    # Copy image
                    try:
                        shutil.copy2(img_file, dest_image)
                    except Exception as e:
                        self.log_error(f"Failed to copy {img_file.name}: {e}")
                        continue
                
                split_stats['images'] += 1
                self.stats['total_images'] += 1
                self.stats['dataset_counts'][dataset_name] += 1
                
                # Remap and copy label
                label_file = labels_dir / f"{img_file.stem}.txt"
                remapped_lines = self.remap_label_file(label_file, class_mapping)
                
                if remapped_lines:
                    if not self.dry_run:
                        try:
                            with open(dest_label, 'w') as f:
                                f.write('\n'.join(remapped_lines))
                        except Exception as e:
                            self.log_error(f"Failed to write label for {new_name}: {e}")
                            continue
                    
                    dataset_annotations += len(remapped_lines)
                    split_stats['annotations'] += len(remapped_lines)
                    self.stats['total_annotations'] += len(remapped_lines)
                else:
                    dataset_skipped += 1
                    split_stats['skipped'] += 1
            
            print(f"  ✓ Processed: {len(image_files)} images, {dataset_annotations} annotations")
            if dataset_skipped > 0:
                print(f"  ⚠️  Skipped: {dataset_skipped} images without valid annotations")
        
        self.stats['split_counts'][split_name] = split_stats['images']
        
        print(f"\n{split_name.upper()} Totals:")
        print(f"  Images: {split_stats['images']:,}")
        print(f"  Annotations: {split_stats['annotations']:,}")
        if split_stats['skipped'] > 0:
            print(f"  Skipped: {split_stats['skipped']:,}")
    
    def create_data_yaml(self):
        """Create merged data.yaml."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.final_classes),
            'names': self.final_classes
        }
        
        if self.dry_run:
            print(f"\n[DRY RUN] Would create data.yaml:")
            print(f"  Path: {self.output_dir / 'data.yaml'}")
            print(f"  Classes: {len(self.final_classes)}")
            for class_id, class_name in sorted(self.final_classes.items()):
                print(f"    {class_id}: {class_name}")
            return
        
        yaml_path = self.output_dir / 'data.yaml'
        
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            print(f"\n✓ Created data.yaml: {yaml_path}")
        except Exception as e:
            self.log_error(f"Failed to create data.yaml: {e}")
    
    def print_summary(self):
        """Print comprehensive merge summary."""
        print(f"\n{'='*70}")
        if self.dry_run:
            print(f"DRY RUN SUMMARY - No files were modified")
        else:
            print(f"MERGE SUMMARY")
        print(f"{'='*70}")
        
        # Class Mapping Summary
        print(f"\nClass Mapping Applied:")
        for dataset_name, dataset_info in self.datasets.items():
            print(f"\n  {dataset_name}:")
            for old_id, new_id in dataset_info['mapping'].items():
                old_name = dataset_info['original_classes'].get(old_id, 'unknown')
                if new_id == -1:
                    print(f"    {old_id}: {old_name} -> SKIPPED")
                else:
                    new_name = self.final_classes[new_id]
                    print(f"    {old_id}: {old_name} -> {new_id}: {new_name}")
        
        # Dataset Statistics
        print(f"\nDatasets Merged: {len([d for d in self.datasets.keys() if self.stats['dataset_counts'][d] > 0])}")
        for name in sorted(self.datasets.keys()):
            count = self.stats['dataset_counts'][name]
            if count > 0:
                print(f"  • {name:<20} {count:>6,} images")
        
        # Split Statistics
        if self.stats['split_counts']:
            print(f"\nSplit Distribution:")
            for split in ['train', 'val', 'test']:
                count = self.stats['split_counts'].get(split, 0)
                if count > 0:
                    pct = count / self.stats['total_images'] * 100
                    print(f"  {split:<6} {count:>6,} images ({pct:>5.1f}%)")
        
        # Overall Statistics
        print(f"\nTotal Statistics:")
        print(f"  Images:      {self.stats['total_images']:>8,}")
        print(f"  Annotations: {self.stats['total_annotations']:>8,}")
        if self.stats['skipped_annotations'] > 0:
            print(f"  Skipped:     {self.stats['skipped_annotations']:>8,} (traffic signs/lights)")
        print(f"  Classes:     {len(self.final_classes):>8}")
        
        # Class Distribution
        if self.stats['class_counts']:
            print(f"\nClass Distribution:")
            total = sum(self.stats['class_counts'].values())
            max_count = max(self.stats['class_counts'].values())
            min_count = min(self.stats['class_counts'].values())
            
            for class_id in sorted(self.final_classes.keys()):
                class_name = self.final_classes[class_id]
                count = self.stats['class_counts'].get(class_name, 0)
                pct = count / total * 100 if total > 0 else 0
                bar_length = int(pct / 2)  # Scale for visualization
                bar = '█' * bar_length
                print(f"  {class_id}. {class_name:<12} {count:>8,} ({pct:>5.1f}%) {bar}")
            
            imbalance_ratio = max_count / min_count if min_count > 0 else 0
            print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}x")
            
            if imbalance_ratio > 50:
                print(f"  ⚠️  HIGH imbalance - consider data augmentation for minority classes")
            elif imbalance_ratio > 20:
                print(f"  ⚠️  MODERATE imbalance - class weights recommended")
            else:
                print(f"  ✓ GOOD class balance")
        
        # Warnings and Errors
        if self.validation_warnings:
            print(f"\n⚠️  Warnings: {len(self.validation_warnings)}")
            for warning in self.validation_warnings[:5]:
                print(f"  - {warning}")
            if len(self.validation_warnings) > 5:
                print(f"  ... and {len(self.validation_warnings) - 5} more")
        
        if self.validation_errors:
            print(f"\n❌ Errors: {len(self.validation_errors)}")
            for error in self.validation_errors[:5]:
                print(f"  - {error}")
            if len(self.validation_errors) > 5:
                print(f"  ... and {len(self.validation_errors) - 5} more")
        
        # Final Status
        print(f"\n{'='*70}")
        if self.dry_run:
            print(f"✓ DRY RUN COMPLETED")
            print(f"{'='*70}")
            print(f"\nTo perform actual merge, run without --dry-run flag")
        elif self.validation_errors:
            print(f"⚠️  MERGE COMPLETED WITH ERRORS")
            print(f"{'='*70}")
            print(f"\nPlease review errors above")
        else:
            print(f"✓ MERGE COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"Output: {self.output_dir}")
            print(f"\nNext steps:")
            print(f"  1. Validate: python scripts/analysis/validate_yolo_dataset.py {self.output_dir}")
            print(f"  2. Analyze: python scripts/analysis/analyze_dataset_balance.py {self.output_dir}")
            print(f"  3. Train: python train.py")


def main():
    parser = argparse.ArgumentParser(
        description='Merge road obstacle detection datasets with validation and class remapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run first (recommended)
  python merge_road_datasets.py --output data/road_obstacles_final --dry-run
  
  # Actual merge
  python merge_road_datasets.py --output data/road_obstacles_final

Final Class Mapping (6 classes):
  0: vehicle     <- cars, car, bus, truck, train, trailer, other_vehicle
  1: person      <- pedestrian, rider, other_person
  2: bicycle     <- bicycle, motorcycle
  3: animal      <- dog-camel-deer, Animals
  4: pothole     <- pothole
  5: debris      <- object (from road_debris)

Skipped Classes (not obstacles):
  - traffic light
  - traffic sign
        """
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview merge without modifying files'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROAD OBSTACLE DATASETS MERGER")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
    print("="*70)
    print("\nFinal Class Mapping (6 classes):")
    print("  0: vehicle     (all vehicle types)")
    print("  1: person      (all person types)")
    print("  2: bicycle     (bicycle + motorcycle)")
    print("  3: animal      (all animals)")
    print("  4: pothole     (road damage)")
    print("  5: debris      (road debris)")
    print(f"\nOutput: {args.output}")
    print("="*70)
    
    try:
        merger = RoadDatasetMerger(args.output, dry_run=args.dry_run)
        
        # Validate datasets first
        if not merger.validate_datasets():
            print("\n❌ Dataset validation failed. Please fix errors before proceeding.")
            return 1
        
        # Create output directory
        if not args.dry_run:
            merger.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Merge all splits
        for split in ['train', 'val', 'test']:
            merger.merge_split(split)
        
        # Create data.yaml
        merger.create_data_yaml()
        
        # Print summary
        merger.print_summary()
        
        if merger.validation_errors:
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n❌ Merge cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
