#!/usr/bin/env python3
"""
XML (Pascal VOC) to COCO Converter - Safety-First Version
Converts Pascal VOC XML annotations to COCO JSON format with validation.

Features:
- Pre-flight validation of input dataset
- Dry-run mode to preview conversion
- Comprehensive error handling
- Split information from splits.json
- Detailed statistics and reporting

Usage:
    # Dry run first (recommended)
    python xml_to_coco.py --input data/pothole_coco --output data/pothole_coco_converted --name pothole --dry-run
    
    # Actual conversion
    python xml_to_coco.py --input data/pothole_coco --output data/pothole_coco_converted --name pothole
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
from collections import defaultdict


class XmlToCocoConverter:
    """Safety-first XML (Pascal VOC) to COCO converter with validation."""
    
    def __init__(self, input_dir: str, output_dir: str, dataset_name: str, dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.dry_run = dry_run
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'missing_images': 0,
            'invalid_xml': 0,
            'copied_images': 0
        }
        
        # Safety checks
        self.validation_errors = []
        self.validation_warnings = []
        
        # COCO structure
        self.category_map = {}  # category_name -> category_id
        self.next_category_id = 0
    
    def log_error(self, message: str):
        """Log an error message."""
        self.validation_errors.append(message)
        print(f"  ❌ ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.validation_warnings.append(message)
        print(f"  ⚠️  WARNING: {message}")
    
    def validate_input_dataset(self) -> bool:
        """
        SAFETY CHECK 1: Validate input dataset structure.
        """
        print(f"\n{'='*70}")
        print(f"SAFETY CHECK 1: Validating Input Dataset")
        print(f"{'='*70}")
        
        # Check if input directory exists
        if not self.input_dir.exists():
            self.log_error(f"Input directory does not exist: {self.input_dir}")
            return False
        
        print(f"✓ Input directory exists: {self.input_dir}")
        
        # Look for annotated-images directory
        annotated_dir = self.input_dir / 'annotated-images'
        if not annotated_dir.exists():
            self.log_error("annotated-images directory not found")
            return False
        
        print(f"✓ Found annotated-images directory")
        
        # Check for splits.json
        splits_file = self.input_dir / 'splits.json'
        if not splits_file.exists():
            self.log_error("splits.json file not found")
            return False
        
        print(f"✓ Found splits.json")
        
        # Load and validate splits.json
        try:
            with open(splits_file, 'r') as f:
                self.splits_data = json.load(f)
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON in splits.json: {e}")
            return False
        except Exception as e:
            self.log_error(f"Failed to read splits.json: {e}")
            return False
        
        # Validate splits structure
        if 'train' not in self.splits_data:
            self.log_error("'train' split not found in splits.json")
            return False
        
        available_splits = [k for k in ['train', 'val', 'test'] if k in self.splits_data]
        print(f"✓ Available splits: {available_splits}")
        
        # Count files
        xml_files = list(annotated_dir.glob('*.xml'))
        jpg_files = list(annotated_dir.glob('*.jpg'))
        
        if not xml_files:
            self.log_error("No XML annotation files found")
            return False
        
        if not jpg_files:
            self.log_error("No JPG image files found")
            return False
        
        print(f"✓ Found {len(xml_files)} XML files")
        print(f"✓ Found {len(jpg_files)} JPG files")
        
        # Sample validation - check a few XML files
        sample_files = xml_files[:5]
        valid_count = 0
        
        for xml_file in sample_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                valid_count += 1
            except Exception as e:
                self.log_warning(f"Invalid XML: {xml_file.name} - {e}")
        
        if valid_count == 0:
            self.log_error("No valid XML files found in sample")
            return False
        
        print(f"✓ XML format validated ({valid_count}/{len(sample_files)} samples)")
        
        print(f"\n{'='*70}")
        print(f"✓ INPUT VALIDATION PASSED")
        print(f"{'='*70}")
        
        return True
    
    def check_output_directory(self) -> bool:
        """
        SAFETY CHECK 2: Check output directory status.
        """
        print(f"\n{'='*70}")
        print(f"SAFETY CHECK 2: Output Directory")
        print(f"{'='*70}")
        
        if self.output_dir.exists():
            # Count existing files
            json_files = list(self.output_dir.rglob('*.json'))
            image_files = list(self.output_dir.rglob('*.jpg'))
            
            if json_files or image_files:
                self.log_warning(f"Output directory already exists with data:")
                print(f"  JSON files: {len(json_files)}")
                print(f"  Images: {len(image_files)}")
                print(f"\n  This data will be OVERWRITTEN during conversion!")
                
                if not self.dry_run:
                    response = input(f"\n  Continue and overwrite? [y/N]: ").strip().lower()
                    if response != 'y':
                        print(f"\n  ❌ Conversion cancelled by user.")
                        return False
            else:
                print(f"✓ Output directory exists but is empty")
        else:
            print(f"✓ Output directory will be created: {self.output_dir}")
        
        return True
    
    def parse_xml_annotation(self, xml_file: Path) -> Optional[Dict]:
        """Parse a single XML annotation file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract basic info
            filename = root.find('filename')
            if filename is None:
                # Use XML filename to infer image filename
                filename = xml_file.stem + '.jpg'
            else:
                filename = filename.text
            
            # Get image size
            size = root.find('size')
            if size is None:
                return None
            
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Extract objects/annotations
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # Get bounding box
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Validate bbox
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    # Clip to image bounds
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)
                
                # Convert to COCO format: [x, y, width, height]
                bbox_coco = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
                
                objects.append({
                    'name': name,
                    'bbox': bbox_coco,
                    'area': area
                })
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except Exception as e:
            return None
    
    def get_or_create_category_id(self, category_name: str) -> int:
        """Get existing category ID or create new one."""
        if category_name not in self.category_map:
            self.category_map[category_name] = self.next_category_id
            self.next_category_id += 1
        return self.category_map[category_name]
    
    def create_coco_structure(self, splits: List[str]):
        """Create COCO directory structure."""
        print(f"\nCreating COCO directory structure...")
        
        if self.dry_run:
            print(f"  [DRY RUN] Would create directories:")
            for split in splits:
                print(f"    - {split}/")
            return
        
        for split in splits:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {split}/")
    
    def convert_split(self, split_name: str, xml_filenames: List[str]) -> Dict:
        """Convert one split to COCO format."""
        print(f"\n{'='*70}")
        print(f"Converting split: {split_name.upper()}")
        if self.dry_run:
            print(f"[DRY RUN MODE - No files will be modified]")
        print(f"{'='*70}")
        
        annotated_dir = self.input_dir / 'annotated-images'
        
        # COCO structure
        coco_data = {
            'info': {
                'description': f'{self.dataset_name} dataset - {split_name} split',
                'version': '1.0',
                'year': 2025,
                'contributor': 'Converted from Pascal VOC XML',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        image_id = 0
        annotation_id = 0
        split_stats = {
            'processed': 0,
            'with_labels': 0,
            'without_labels': 0,
            'invalid_xml': 0,
            'copied_images': 0,
            'missing_images': 0
        }
        
        print(f"\nProcessing {len(xml_filenames)} files...")
        
        for xml_filename in xml_filenames:
            xml_path = annotated_dir / xml_filename
            
            if not xml_path.exists():
                split_stats['missing_images'] += 1
                if split_stats['missing_images'] <= 3:
                    self.log_warning(f"XML file not found: {xml_filename}")
                continue
            
            # Parse XML
            annotation = self.parse_xml_annotation(xml_path)
            
            if annotation is None:
                split_stats['invalid_xml'] += 1
                if split_stats['invalid_xml'] <= 3:
                    self.log_warning(f"Failed to parse XML: {xml_filename}")
                continue
            
            # Check if image file exists
            img_filename = annotation['filename']
            img_path = annotated_dir / img_filename
            
            if not img_path.exists():
                split_stats['missing_images'] += 1
                if split_stats['missing_images'] <= 3:
                    self.log_warning(f"Image file not found: {img_filename}")
                continue
            
            # Add image to COCO
            coco_data['images'].append({
                'id': image_id,
                'file_name': img_filename,
                'width': annotation['width'],
                'height': annotation['height']
            })
            
            # Copy image file
            if not self.dry_run:
                dst_img = self.output_dir / split_name / img_filename
                try:
                    shutil.copy2(img_path, dst_img)
                    split_stats['copied_images'] += 1
                except Exception as e:
                    self.log_error(f"Failed to copy {img_filename}: {e}")
                    continue
            else:
                split_stats['copied_images'] += 1
            
            # Add annotations
            if annotation['objects']:
                for obj in annotation['objects']:
                    category_name = obj['name']
                    category_id = self.get_or_create_category_id(category_name)
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': obj['bbox'],
                        'area': obj['area'],
                        'iscrowd': 0
                    })
                    annotation_id += 1
                
                split_stats['with_labels'] += 1
            else:
                split_stats['without_labels'] += 1
            
            image_id += 1
            split_stats['processed'] += 1
            
            # Progress update
            if split_stats['processed'] % 100 == 0:
                print(f"  Processed: {split_stats['processed']}/{len(xml_filenames)}")
        
        # Add categories to COCO data
        for cat_name, cat_id in sorted(self.category_map.items(), key=lambda x: x[1]):
            coco_data['categories'].append({
                'id': cat_id,
                'name': cat_name,
                'supercategory': 'object'
            })
        
        # Write COCO JSON file
        if not self.dry_run:
            json_path = self.output_dir / split_name / '_annotations.coco.json'
            try:
                with open(json_path, 'w') as f:
                    json.dump(coco_data, f, indent=2)
                print(f"\n✓ Created COCO JSON: {json_path.name}")
            except Exception as e:
                self.log_error(f"Failed to write COCO JSON: {e}")
                return None
        else:
            print(f"\n[DRY RUN] Would create: {split_name}/_annotations.coco.json")
        
        # Print statistics
        print(f"\n{split_name.upper()} Statistics:")
        print(f"  ✓ Images processed: {split_stats['processed']}")
        print(f"  ✓ Images copied: {split_stats['copied_images']}")
        print(f"  ✓ Images with labels: {split_stats['with_labels']}")
        print(f"  ✓ Images without labels: {split_stats['without_labels']}")
        print(f"  ✓ Total annotations: {len(coco_data['annotations'])}")
        print(f"  ✓ Categories: {len(coco_data['categories'])}")
        
        if split_stats['missing_images'] > 0:
            print(f"  ⚠️  Missing files: {split_stats['missing_images']}")
        if split_stats['invalid_xml'] > 0:
            print(f"  ⚠️  Invalid XML: {split_stats['invalid_xml']}")
        
        # Update global stats
        self.stats['total_images'] += split_stats['processed']
        self.stats['total_annotations'] += len(coco_data['annotations'])
        self.stats['copied_images'] += split_stats['copied_images']
        self.stats['missing_images'] += split_stats['missing_images']
        self.stats['invalid_xml'] += split_stats['invalid_xml']
        
        return coco_data
    
    def convert(self) -> bool:
        """Main conversion process with safety checks."""
        print(f"\n{'='*70}")
        print(f"XML (Pascal VOC) to COCO Conversion: {self.dataset_name}")
        if self.dry_run:
            print(f"[DRY RUN MODE - No files will be modified]")
        print(f"{'='*70}")
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        
        # SAFETY CHECK 1: Validate input
        if not self.validate_input_dataset():
            print(f"\n{'='*70}")
            print(f"❌ INPUT VALIDATION FAILED")
            print(f"{'='*70}")
            return False
        
        # SAFETY CHECK 2: Check output
        if not self.check_output_directory():
            return False
        
        # Get available splits
        available_splits = [k for k in ['train', 'val', 'test'] if k in self.splits_data]
        
        # Create directory structure
        self.create_coco_structure(available_splits)
        
        # Convert each split
        for split in available_splits:
            xml_filenames = self.splits_data[split]
            result = self.convert_split(split, xml_filenames)
            
            if result is None and not self.dry_run:
                self.log_error(f"Failed to convert {split} split")
                return False
        
        # Print category mapping
        print(f"\n{'='*70}")
        print(f"Category Mapping:")
        print(f"{'='*70}")
        for cat_name, cat_id in sorted(self.category_map.items(), key=lambda x: x[1]):
            print(f"  {cat_id}: {cat_name}")
        
        # Final report
        print(f"\n{'='*70}")
        if self.dry_run:
            print(f"✓ DRY RUN COMPLETED - No files were modified")
        else:
            print(f"✓ CONVERSION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        
        print(f"\nFinal Statistics:")
        print(f"  Images processed: {self.stats['total_images']}")
        print(f"  Annotations created: {self.stats['total_annotations']}")
        print(f"  Images copied: {self.stats['copied_images']}")
        print(f"  Missing files: {self.stats['missing_images']}")
        print(f"  Invalid XML: {self.stats['invalid_xml']}")
        print(f"  Categories: {len(self.category_map)}")
        
        if self.validation_warnings:
            print(f"\n⚠️  Warnings: {len(self.validation_warnings)}")
        
        if self.validation_errors:
            print(f"\n❌ Errors encountered: {len(self.validation_errors)}")
            return False
        
        if not self.dry_run:
            print(f"\nOutput directory: {self.output_dir}")
            print(f"\nNext step: Convert to YOLO format:")
            print(f"  python scripts/preprocessing/coco_to_yolo.py --input {self.output_dir} --output {self.output_dir.parent / (self.dataset_name + '_yolo')} --name {self.dataset_name}")
        else:
            print(f"\nTo perform actual conversion, run without --dry-run flag")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Pascal VOC XML annotations to COCO JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run first (recommended)
  python xml_to_coco.py --input data/pothole_coco --output data/pothole_coco_converted --name pothole --dry-run
  
  # Actual conversion
  python xml_to_coco.py --input data/pothole_coco --output data/pothole_coco_converted --name pothole

Structure:
  Input should have:
    - annotated-images/ (with .xml and .jpg files)
    - splits.json (with train/val/test split definitions)
  
  Output will have COCO format:
    - train/_annotations.coco.json + images
    - val/_annotations.coco.json + images (if available)
    - test/_annotations.coco.json + images (if available)
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input dataset directory with XML annotations')
    parser.add_argument('--output', '-o', required=True, help='Output COCO format directory')
    parser.add_argument('--name', '-n', required=True, help='Dataset name')
    parser.add_argument('--dry-run', action='store_true', help='Preview conversion without modifying files')
    
    args = parser.parse_args()
    
    try:
        converter = XmlToCocoConverter(args.input, args.output, args.name, args.dry_run)
        success = converter.convert()
        
        if not success:
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n❌ Conversion cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
