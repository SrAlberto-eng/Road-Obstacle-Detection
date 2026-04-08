#!/usr/bin/env python3
"""
Extract and Organize BDD100K Dataset
Creates a clean BDD100K dataset with only matching images and labels,
organized in train/val/test splits.

Usage:
    python extract_bdd100k_labels.py --images data/bdd100k/images/10k --labels data/bdd100k/labels/100k --output data/bdd100k_clean
"""

import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def extract_labels_for_images(images_base: Path, labels_base: Path, output_base: Path):
    """Extract only labels that have corresponding images and organize dataset."""
    
    print("="*70)
    print("BDD100K DATASET ORGANIZER")
    print("="*70)
    print(f"Images:  {images_base}")
    print(f"Labels:  {labels_base}")
    print(f"Output:  {output_base}")
    print("="*70)
    
    stats = {
        'total_images': 0,
        'images_copied': 0,
        'labels_found': 0,
        'labels_not_found': 0
    }
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing: {split_name.upper()}")
        print(f"{'='*70}")
        
        images_dir = images_base / split_name
        labels_dir = labels_base / split_name
        
        # Create output directories for both images and labels
        output_images_dir = output_base / 'images' / split_name
        output_labels_dir = output_base / 'labels' / split_name
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if directories exist
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
        
        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            continue
        
        # Get all images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"Found {len(image_files):,} images")
        
        split_stats = {
            'images': len(image_files),
            'found': 0,
            'not_found': 0
        }
        
        # For each image, try to find corresponding label
        for img_file in image_files:
            # Get image name without extension
            img_name = img_file.stem
            
            # Try to find label file
            label_file = labels_dir / f"{img_name}.json"
            
            if label_file.exists():
                # Copy image to output
                dest_image = output_images_dir / img_file.name
                shutil.copy2(img_file, dest_image)
                
                # Copy label to output
                dest_label = output_labels_dir / label_file.name
                shutil.copy2(label_file, dest_label)
                
                split_stats['found'] += 1
                stats['labels_found'] += 1
                stats['images_copied'] += 1
            else:
                split_stats['not_found'] += 1
                stats['labels_not_found'] += 1
                if split_stats['not_found'] <= 5:  # Show first 5
                    print(f"  [WARNING] Label not found for: {img_name}")
        
        stats['total_images'] += split_stats['images']
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Images processed: {split_stats['images']:,}")
        print(f"  Pairs copied: {split_stats['found']:,} ({split_stats['found']/split_stats['images']*100:.1f}%)")
        print(f"  Missing labels: {split_stats['not_found']:,} ({split_stats['not_found']/split_stats['images']*100:.1f}%)")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"ORGANIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {stats['total_images']:,}")
    print(f"Images + Labels copied: {stats['images_copied']:,} ({stats['images_copied']/stats['total_images']*100:.1f}%)")
    print(f"Missing labels: {stats['labels_not_found']:,} ({stats['labels_not_found']/stats['total_images']*100:.1f}%)")
    print(f"\n✓ Organized dataset saved to: {output_base}")
    print(f"\nDataset structure:")
    print(f"  {output_base}/images/train/")
    print(f"  {output_base}/images/val/")
    print(f"  {output_base}/images/test/")
    print(f"  {output_base}/labels/train/")
    print(f"  {output_base}/labels/val/")
    print(f"  {output_base}/labels/test/")
    print(f"\nNext step:")
    print(f"  Convert to YOLO:")
    print(f"    python scripts/bdd100k_to_yolo_direct.py --input {output_base} --output data/bdd100k_yolo")


def main():
    parser = argparse.ArgumentParser(
        description='Extract and organize BDD100K dataset with matching images and labels'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to images directory (e.g., data/bdd100k/images/10k)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to labels directory (e.g., data/bdd100k/labels/100k)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized dataset (e.g., data/bdd100k_clean)'
    )
    
    args = parser.parse_args()
    
    images_base = Path(args.images)
    labels_base = Path(args.labels)
    output_base = Path(args.output)
    
    if not images_base.exists():
        print(f"Error: Images directory not found: {images_base}")
        return 1
    
    if not labels_base.exists():
        print(f"Error: Labels directory not found: {labels_base}")
        return 1
    
    extract_labels_for_images(images_base, labels_base, output_base)
    return 0


if __name__ == '__main__':
    exit(main())
