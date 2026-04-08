#!/usr/bin/env python3
"""
YOLO Dataset Validator - Simple & Clear
Validates YOLO format dataset structure and data integrity.

Usage:
    python validate_yolo_dataset.py --dataset data/animals_yolo
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class YoloValidator:
    """Simple validator for YOLO format datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.issues = []
        self.stats = {}
    
    def log_issue(self, severity: str, message: str):
        """Log a validation issue."""
        self.issues.append({'severity': severity, 'message': message})
        icon = '❌' if severity == 'ERROR' else '⚠'
        print(f"{icon} {severity}: {message}")
    
    def validate_structure(self) -> bool:
        """Validate basic directory structure."""
        print("\n" + "="*60)
        print("1. Validating Directory Structure")
        print("="*60)
        
        # Check root exists
        if not self.dataset_path.exists():
            self.log_issue('ERROR', f"Dataset path does not exist: {self.dataset_path}")
            return False
        
        print(f"✓ Dataset root exists: {self.dataset_path}")
        
        # Check data.yaml
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            self.log_issue('ERROR', "Missing data.yaml configuration file")
            return False
        
        print(f"✓ data.yaml found")
        
        # Check images and labels directories
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self.log_issue('ERROR', f"Missing required directory: {dir_name}/")
                return False
            print(f"✓ {dir_name}/ directory exists")
        
        return True
    
    def validate_data_yaml(self) -> Dict:
        """Validate and parse data.yaml."""
        print("\n" + "="*60)
        print("2. Validating data.yaml")
        print("="*60)
        
        yaml_path = self.dataset_path / 'data.yaml'
        
        # Simple YAML parser (only for basic key: value format)
        config = {}
        class_names = {}
        
        with open(yaml_path, 'r') as f:
            in_names_section = False
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                if line == 'names:':
                    in_names_section = True
                    continue
                
                if in_names_section:
                    # Parse class definition: "  0: dog-camel-deer"
                    if ':' in line:
                        parts = line.split(':', 1)
                        class_id = int(parts[0].strip())
                        class_name = parts[1].strip()
                        class_names[class_id] = class_name
                elif ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
        
        # Validate required fields
        required_fields = ['train', 'val']
        for field in required_fields:
            if field not in config:
                self.log_issue('ERROR', f"Missing required field in data.yaml: {field}")
                return None
        
        print(f"✓ train split: {config['train']}")
        print(f"✓ val split: {config['val']}")
        
        if 'test' in config:
            print(f"✓ test split: {config['test']}")
        
        if not class_names:
            self.log_issue('ERROR', "No classes defined in data.yaml")
            return None
        
        print(f"\n✓ Found {len(class_names)} classes:")
        for class_id, class_name in sorted(class_names.items()):
            print(f"    {class_id}: {class_name}")
        
        config['class_names'] = class_names
        return config
    
    def validate_split(self, split_name: str) -> Tuple[int, int]:
        """Validate one split (train/val/test)."""
        print(f"\n{'-'*60}")
        print(f"Validating split: {split_name}")
        print(f"{'-'*60}")
        
        img_dir = self.dataset_path / 'images' / split_name
        lbl_dir = self.dataset_path / 'labels' / split_name
        
        # Check directories exist
        if not img_dir.exists():
            self.log_issue('ERROR', f"Missing directory: images/{split_name}/")
            return 0, 0
        
        if not lbl_dir.exists():
            self.log_issue('ERROR', f"Missing directory: labels/{split_name}/")
            return 0, 0
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(img_dir.glob(f'*{ext}')))
            image_files.extend(list(img_dir.glob(f'*{ext.upper()}')))
        
        if not image_files:
            self.log_issue('WARNING', f"No images found in images/{split_name}/")
            return 0, 0
        
        print(f"  Found {len(image_files)} images")
        
        # Check corresponding label files
        missing_labels = []
        invalid_labels = []
        total_objects = 0
        
        for img_path in image_files:
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            
            # Check label file exists
            if not lbl_path.exists():
                missing_labels.append(img_path.name)
                continue
            
            # Validate label file content
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_labels.append(f"{lbl_path.name}:{line_num} - Expected 5 values, got {len(parts)}")
                        continue
                    
                    # Validate format: class_id x_center y_center width height
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= x_center <= 1):
                            invalid_labels.append(f"{lbl_path.name}:{line_num} - x_center out of range [0,1]: {x_center}")
                        if not (0 <= y_center <= 1):
                            invalid_labels.append(f"{lbl_path.name}:{line_num} - y_center out of range [0,1]: {y_center}")
                        if not (0 < width <= 1):
                            invalid_labels.append(f"{lbl_path.name}:{line_num} - width out of range (0,1]: {width}")
                        if not (0 < height <= 1):
                            invalid_labels.append(f"{lbl_path.name}:{line_num} - height out of range (0,1]: {height}")
                        
                        total_objects += 1
                        
                    except ValueError as e:
                        invalid_labels.append(f"{lbl_path.name}:{line_num} - Invalid number format: {e}")
                
            except Exception as e:
                invalid_labels.append(f"{lbl_path.name} - Error reading file: {e}")
        
        # Report results
        print(f"  ✓ Total objects: {total_objects}")
        
        if missing_labels:
            self.log_issue('WARNING', f"Found {len(missing_labels)} images without label files")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    print(f"      - {name}")
        
        if invalid_labels:
            self.log_issue('ERROR', f"Found {len(invalid_labels)} invalid label entries")
            # Show first 10 invalid labels
            for issue in invalid_labels[:10]:
                print(f"      - {issue}")
            if len(invalid_labels) > 10:
                print(f"      ... and {len(invalid_labels) - 10} more")
        
        if not missing_labels and not invalid_labels:
            print(f"  ✓ All labels valid")
        
        return len(image_files), total_objects
    
    def validate(self) -> bool:
        """Run complete validation."""
        print("\n" + "="*60)
        print("YOLO DATASET VALIDATION")
        print("="*60)
        print(f"Dataset: {self.dataset_path}")
        
        # Step 1: Structure
        if not self.validate_structure():
            print("\n❌ VALIDATION FAILED: Invalid structure")
            return False
        
        # Step 2: data.yaml
        config = self.validate_data_yaml()
        if config is None:
            print("\n❌ VALIDATION FAILED: Invalid data.yaml")
            return False
        
        # Step 3: Validate each split
        print("\n" + "="*60)
        print("3. Validating Splits")
        print("="*60)
        
        splits = ['train', 'val']
        if 'test' in config:
            splits.append('test')
        
        total_images = 0
        total_objects = 0
        
        for split in splits:
            num_images, num_objects = self.validate_split(split)
            total_images += num_images
            total_objects += num_objects
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        print(f"\nDataset Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Total objects: {total_objects}")
        print(f"  Classes: {len(config['class_names'])}")
        print(f"  Splits: {', '.join(splits)}")
        
        # Issues summary
        errors = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        
        if errors:
            print(f"\n❌ Found {len(errors)} errors")
            print("   Dataset has critical issues that must be fixed!")
            return False
        
        if warnings:
            print(f"\n⚠ Found {len(warnings)} warnings")
            print("  Dataset is valid but has minor issues")
        else:
            print("\n✅ No issues found!")
        
        print("\n✓ VALIDATION PASSED")
        print(f"\nDataset is ready for training!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_yolo_dataset.py data/animals_yolo
  python validate_yolo_dataset.py data/road_obstacles_final
        """
    )
    
    parser.add_argument('dataset', help='Path to YOLO dataset directory')
    
    args = parser.parse_args()
    
    try:
        validator = YoloValidator(args.dataset)
        success = validator.validate()
        return 0 if success else 1
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
