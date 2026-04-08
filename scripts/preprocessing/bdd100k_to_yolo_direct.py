#!/usr/bin/env python3
"""
BDD100K to YOLO Converter - Direct and Simple
Converts BDD100K format directly to YOLO format without intermediate COCO step.

Focuses on road obstacles:
- vehicle (car, truck, bus)
- person (pedestrians)
- two_wheeler (bike, motor, rider)

Usage:
    python bdd100k_to_yolo_direct.py --input data/bdd100k --output data/bdd100k_yolo --sample 10000
"""

import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional
from PIL import Image


class BDD100KToYOLOConverter:
    """Convert BDD100K directly to YOLO format."""
    
    def __init__(self, input_dir: str, output_dir: str, max_images: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_images = max_images
        
        # Class mapping for road obstacles
        self.class_mapping = {
            'car': 'vehicle',
            'truck': 'vehicle',
            'bus': 'vehicle',
            'person': 'person',
            'bike': 'two_wheeler',
            'motor': 'two_wheeler',
            'rider': 'two_wheeler',
        }
        
        # YOLO class IDs
        self.yolo_classes = {
            'vehicle': 0,
            'person': 1,
            'two_wheeler': 2
        }
        
        self.stats = {
            'total_json_files': 0,
            'images_found': 0,
            'images_not_found': 0,
            'images_with_obstacles': 0,
            'images_without_obstacles': 0,
            'total_annotations': 0,
            'class_counts': Counter()
        }
    
    def convert_bbox_to_yolo(self, box2d: Dict, img_width: int, img_height: int) -> Optional[List[float]]:
        """
        Convert BDD100K box2d to YOLO format.
        BDD: {x1, y1, x2, y2} (absolute pixels)
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
        """
        x1 = float(box2d['x1'])
        y1 = float(box2d['y1'])
        x2 = float(box2d['x2'])
        y2 = float(box2d['y2'])
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Validate
        if width <= 0 or height <= 0:
            return None
        
        # Calculate center
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    
    def find_image_file(self, image_name: str, images_dir: Path) -> Optional[Path]:
        """Find image file with various possible extensions and naming conventions."""
        # Try exact name first
        for ext in ['.jpg', '.png', '.jpeg']:
            # With extension in name
            img_path = images_dir / f"{image_name}{ext}"
            if img_path.exists():
                return img_path
            
            # Without extension (might already be in name)
            img_path = images_dir / image_name
            if img_path.exists():
                return img_path
        
        return None
    
    def convert_split(self, split_name: str) -> bool:
        """Convert one split (train/val/test)."""
        print(f"\n{'='*70}")
        print(f"Converting: {split_name.upper()}")
        print(f"{'='*70}")
        
        # Paths - Support both old and new structure
        # Try new structure first (bdd100k_clean)
        labels_dir = self.input_dir / 'labels' / split_name
        images_dir = self.input_dir / 'images' / split_name
        
        # Fall back to old structure if needed
        if not labels_dir.exists():
            labels_dir = self.input_dir / 'labels' / '100k' / split_name
        if not images_dir.exists():
            images_dir = self.input_dir / 'images' / '10k' / split_name
        
        # Check if directories exist
        if not labels_dir.exists():
            print(f"[WARNING] Labels directory not found: {labels_dir}")
            return False
        
        if not images_dir.exists():
            print(f"[WARNING] Images directory not found: {images_dir}")
            return False
        
        # Create output directories
        yolo_split_name = 'train' if split_name == 'train' else 'val' if split_name == 'val' else 'test'
        output_images = self.output_dir / 'images' / yolo_split_name
        output_labels = self.output_dir / 'labels' / yolo_split_name
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Get label files
        json_files = sorted(labels_dir.glob('*.json'))
        self.stats['total_json_files'] += len(json_files)
        
        print(f"Found {len(json_files):,} label files")
        
        if self.max_images and split_name == 'train':
            json_files = json_files[:self.max_images]
            print(f"Limiting to {self.max_images:,} images for training")
        
        split_stats = {
            'processed': 0,
            'found': 0,
            'not_found': 0,
            'with_obstacles': 0,
            'without_obstacles': 0,
            'annotations': 0
        }
        
        # Process each label file
        for idx, json_file in enumerate(json_files, 1):
            if idx % 1000 == 0:
                print(f"  Progress: {idx:,}/{len(json_files):,} ({idx/len(json_files)*100:.1f}%)")
            
            try:
                # Load BDD100K annotation
                with open(json_file, 'r', encoding='utf-8') as f:
                    bdd_data = json.load(f)
                
                image_name = bdd_data.get('name', json_file.stem)
                
                # Find image file
                image_file = self.find_image_file(image_name, images_dir)
                
                if not image_file:
                    split_stats['not_found'] += 1
                    self.stats['images_not_found'] += 1
                    if idx <= 10:  # Show first 10 missing images
                        print(f"  [WARNING] Image not found: {image_name}")
                    continue
                
                split_stats['found'] += 1
                self.stats['images_found'] += 1
                
                # Get image dimensions
                try:
                    with Image.open(image_file) as img:
                        img_width, img_height = img.size
                except:
                    # Fallback to BDD100K standard resolution
                    img_width, img_height = 1280, 720
                
                # Process annotations
                yolo_annotations = []
                
                # BDD100K has 'labels' or 'frames' structure
                objects = []
                if 'labels' in bdd_data:
                    objects = bdd_data['labels']
                elif 'frames' in bdd_data:
                    for frame in bdd_data['frames']:
                        objects.extend(frame.get('objects', []))
                
                for obj in objects:
                    # Check if it's a detection object
                    if 'box2d' not in obj or obj['box2d'] is None:
                        continue
                    
                    category = obj.get('category', '')
                    
                    # Check if this is an obstacle class
                    if category not in self.class_mapping:
                        continue
                    
                    # Get mapped class
                    mapped_class = self.class_mapping[category]
                    class_id = self.yolo_classes[mapped_class]
                    
                    # Convert bbox
                    yolo_bbox = self.convert_bbox_to_yolo(obj['box2d'], img_width, img_height)
                    
                    if yolo_bbox is None:
                        continue
                    
                    # Add annotation
                    yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
                    split_stats['annotations'] += 1
                    self.stats['total_annotations'] += 1
                    self.stats['class_counts'][mapped_class] += 1
                
                # Copy image and save label
                dest_image = output_images / image_file.name
                shutil.copy2(image_file, dest_image)
                
                # Save YOLO label (even if empty)
                label_file = output_labels / f"{image_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                if yolo_annotations:
                    split_stats['with_obstacles'] += 1
                    self.stats['images_with_obstacles'] += 1
                else:
                    split_stats['without_obstacles'] += 1
                    self.stats['images_without_obstacles'] += 1
                
                split_stats['processed'] += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed to process {json_file.name}: {e}")
        
        # Print split statistics
        print(f"\n{split_name.upper()} Results:")
        print(f"  Images processed: {split_stats['processed']:,}")
        print(f"  Images found: {split_stats['found']:,}")
        print(f"  Images not found: {split_stats['not_found']:,}")
        print(f"  Images with obstacles: {split_stats['with_obstacles']:,}")
        print(f"  Images without obstacles: {split_stats['without_obstacles']:,}")
        print(f"  Total annotations: {split_stats['annotations']:,}")
        
        return split_stats['processed'] > 0
    
    def create_data_yaml(self):
        """Create YOLO data.yaml file."""
        yaml_path = self.output_dir / 'data.yaml'
        
        yaml_content = f"""# BDD100K Road Obstacles Dataset (YOLO Format)
# Converted from BDD100K with class merging for road obstacle detection

path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 3
names:
  0: vehicle
  1: person
  2: two_wheeler

# Class mapping applied:
# - vehicle: car, truck, bus (large obstacles)
# - person: pedestrian (vulnerable road user)
# - two_wheeler: bike, motor, rider (vulnerable vehicles)
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Created data.yaml: {yaml_path}")
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"CONVERSION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total JSON files: {self.stats['total_json_files']:,}")
        print(f"  Images found: {self.stats['images_found']:,}")
        print(f"  Images not found: {self.stats['images_not_found']:,}")
        print(f"  Images with obstacles: {self.stats['images_with_obstacles']:,}")
        print(f"  Images without obstacles: {self.stats['images_without_obstacles']:,}")
        print(f"  Total annotations: {self.stats['total_annotations']:,}")
        
        if self.stats['class_counts']:
            print(f"\nClass Distribution:")
            total = sum(self.stats['class_counts'].values())
            for class_name, count in self.stats['class_counts'].most_common():
                pct = count / total * 100
                print(f"  {class_name:<15} {count:>8,} ({pct:>6.2f}%)")
            
            # Calculate imbalance
            counts = list(self.stats['class_counts'].values())
            max_count = max(counts)
            min_count = min(counts)
            ratio = max_count / min_count
            
            print(f"\n  Imbalance ratio: {ratio:.1f}x")
            if ratio > 50:
                print(f"  [WARNING] SEVERE imbalance - use class weights")
            elif ratio > 20:
                print(f"  [WARNING] MODERATE imbalance - class weights recommended")
            else:
                print(f"  [OK] Acceptable balance")
        
        print(f"\n{'='*70}")
        print(f"✓ CONVERSION COMPLETED")
        print(f"{'='*70}")
        print(f"Output: {self.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Validate: python scripts/validate_yolo_dataset.py --dataset {self.output_dir}")
        print(f"  2. Train: yolo train data={self.output_dir}/data.yaml model=yolov8n.pt epochs=100")


def main():
    parser = argparse.ArgumentParser(
        description='Convert BDD100K dataset directly to YOLO format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to BDD100K root directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Limit training set to N images (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("BDD100K to YOLO Converter")
    print("="*70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.sample:
        print(f"Sample: {args.sample:,} training images")
    print("="*70)
    
    converter = BDD100KToYOLOConverter(args.input, args.output, args.sample)
    
    # Convert all splits
    splits = ['train', 'val', 'test']
    success = False
    
    for split in splits:
        if converter.convert_split(split):
            success = True
    
    if success:
        converter.create_data_yaml()
        converter.print_summary()
        return 0
    else:
        print("\n[ERROR] No data converted. Check input paths.")
        return 1


if __name__ == '__main__':
    exit(main())
