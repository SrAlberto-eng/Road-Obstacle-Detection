#!/usr/bin/env python3
"""
Dataset Balance Analyzer
Analyzes class distribution in YOLO format datasets to detect imbalance.

Usage:
    python analyze_dataset_balance.py --dataset data/animals_yolo
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json


class DatasetBalanceAnalyzer:
    """Analyze class distribution and balance in YOLO datasets."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.class_names = {}
        self.class_counts = defaultdict(lambda: defaultdict(int))
        self.image_counts = defaultdict(int)
        self.images_per_class = defaultdict(lambda: defaultdict(set))
    
    def load_class_names(self):
        """Load class names from data.yaml."""
        yaml_path = self.dataset_path / 'data.yaml'
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_path}")
        
        with open(yaml_path, 'r') as f:
            in_names_section = False
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if line == 'names:':
                    in_names_section = True
                    continue
                
                if in_names_section and ':' in line:
                    parts = line.split(':', 1)
                    class_id = int(parts[0].strip())
                    class_name = parts[1].strip()
                    self.class_names[class_id] = class_name
        
        return self.class_names
    
    def analyze_split(self, split_name: str):
        """Analyze class distribution in one split."""
        labels_dir = self.dataset_path / 'labels' / split_name
        
        if not labels_dir.exists():
            return
        
        label_files = list(labels_dir.glob('*.txt'))
        self.image_counts[split_name] = len(label_files)
        
        for label_file in label_files:
            image_name = label_file.stem
            
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        self.class_counts[split_name][class_id] += 1
                        self.images_per_class[split_name][class_id].add(image_name)
    
    def calculate_imbalance_ratio(self, counts):
        """Calculate imbalance ratio (max/min)."""
        if not counts or len(counts) < 2:
            return 1.0
        
        values = list(counts.values())
        max_count = max(values)
        min_count = min(values)
        
        if min_count == 0:
            return float('inf')
        
        return max_count / min_count
    
    def print_split_analysis(self, split_name: str):
        """Print detailed analysis for one split."""
        if split_name not in self.class_counts:
            return
        
        print(f"\n{'='*70}")
        print(f"Split: {split_name.upper()}")
        print(f"{'='*70}")
        
        total_images = self.image_counts[split_name]
        total_objects = sum(self.class_counts[split_name].values())
        
        print(f"\nOverview:")
        print(f"  Total images: {total_images}")
        print(f"  Total objects: {total_objects}")
        print(f"  Avg objects per image: {total_objects / total_images:.2f}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        print(f"  {'Class ID':<10} {'Class Name':<25} {'Objects':<10} {'% Total':<10} {'Images':<10} {'% Images'}")
        print(f"  {'-'*10} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        class_stats = []
        for class_id in sorted(self.class_names.keys()):
            class_name = self.class_names[class_id]
            obj_count = self.class_counts[split_name].get(class_id, 0)
            img_count = len(self.images_per_class[split_name].get(class_id, set()))
            
            obj_pct = (obj_count / total_objects * 100) if total_objects > 0 else 0
            img_pct = (img_count / total_images * 100) if total_images > 0 else 0
            
            print(f"  {class_id:<10} {class_name:<25} {obj_count:<10} {obj_pct:<10.2f} {img_count:<10} {img_pct:.2f}%")
            
            class_stats.append({
                'class_id': class_id,
                'class_name': class_name,
                'objects': obj_count,
                'images': img_count,
                'obj_pct': obj_pct,
                'img_pct': img_pct
            })
        
        # Imbalance analysis
        imbalance_ratio = self.calculate_imbalance_ratio(self.class_counts[split_name])
        
        print(f"\nBalance Analysis:")
        print(f"  Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio == float('inf'):
            print(f"  ⚠️  CRITICAL: At least one class has ZERO instances!")
        elif imbalance_ratio > 10:
            print(f"  ⚠️  WARNING: Severe class imbalance detected!")
            print(f"      - Ratio > 10 indicates one class is 10x more common than another")
            print(f"      - Consider: oversampling minority classes, class weights, or focal loss")
        elif imbalance_ratio > 3:
            print(f"  ⚠️  MODERATE: Noticeable class imbalance")
            print(f"      - Ratio > 3 may cause model bias")
            print(f"      - Consider: using class weights during training")
        else:
            print(f"  ✅ GOOD: Classes are reasonably balanced")
        
        return class_stats, imbalance_ratio
    
    def print_recommendations(self, all_stats):
        """Print recommendations based on analysis."""
        print(f"\n{'='*70}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*70}")
        
        # Check if any split has severe imbalance
        severe_imbalance = any(stats['imbalance'] > 10 for stats in all_stats.values())
        moderate_imbalance = any(3 < stats['imbalance'] <= 10 for stats in all_stats.values())
        
        if severe_imbalance:
            print(f"\n⚠️  SEVERE IMBALANCE DETECTED")
            print(f"\nRecommended Actions:")
            print(f"  1. Use class weights in loss function:")
            print(f"     - Calculate weights inversely proportional to class frequency")
            print(f"     - weight[c] = total_samples / (num_classes * samples_per_class[c])")
            print(f"\n  2. Apply data augmentation to minority classes:")
            print(f"     - Increase augmentation probability for underrepresented classes")
            print(f"     - Consider SMOTE or other oversampling techniques")
            print(f"\n  3. Use focal loss:")
            print(f"     - Helps model focus on hard-to-classify examples")
            print(f"     - Particularly effective for severe imbalance")
            print(f"\n  4. Collect more data:")
            print(f"     - Focus on underrepresented classes")
            print(f"     - This is often the best long-term solution")
        
        elif moderate_imbalance:
            print(f"\n⚠️  MODERATE IMBALANCE DETECTED")
            print(f"\nRecommended Actions:")
            print(f"  1. Use class weights in training:")
            print(f"     - Most frameworks support this (YOLOv8, PyTorch, etc.)")
            print(f"\n  2. Monitor per-class metrics:")
            print(f"     - Track precision, recall, mAP for each class separately")
            print(f"     - Ensure minority classes aren't being ignored")
            print(f"\n  3. Consider slight augmentation boost:")
            print(f"     - Apply more augmentation to minority classes")
        
        else:
            print(f"\n✅ BALANCED DATASET")
            print(f"\nDataset appears well-balanced. You can proceed with standard training.")
            print(f"\nBest Practices:")
            print(f"  1. Still monitor per-class metrics during training")
            print(f"  2. Use validation set to check for overfitting")
            print(f"  3. Consider stratified splits if combining datasets")
        
        # Calculate suggested class weights
        print(f"\n{'='*70}")
        print(f"SUGGESTED CLASS WEIGHTS")
        print(f"{'='*70}")
        
        for split_name, stats in all_stats.items():
            if stats['class_stats']:
                print(f"\n{split_name.upper()} Split:")
                
                total_objects = sum(s['objects'] for s in stats['class_stats'])
                num_classes = len(stats['class_stats'])
                
                if total_objects > 0 and num_classes > 0:
                    print(f"  {'Class':<25} {'Weight':<10} {'Formula'}")
                    print(f"  {'-'*25} {'-'*10} {'-'*40}")
                    
                    for class_stat in stats['class_stats']:
                        if class_stat['objects'] > 0:
                            weight = total_objects / (num_classes * class_stat['objects'])
                            formula = f"{total_objects} / ({num_classes} * {class_stat['objects']})"
                            print(f"  {class_stat['class_name']:<25} {weight:<10.3f} {formula}")
                        else:
                            print(f"  {class_stat['class_name']:<25} {'N/A':<10} (no samples)")
    
    def analyze(self):
        """Run complete analysis."""
        print(f"\n{'='*70}")
        print(f"DATASET BALANCE ANALYSIS")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_path}")
        
        # Load class names
        self.load_class_names()
        print(f"\nClasses Found: {len(self.class_names)}")
        for class_id, class_name in sorted(self.class_names.items()):
            print(f"  {class_id}: {class_name}")
        
        # Analyze each split
        splits = ['train', 'val', 'test']
        all_stats = {}
        
        for split in splits:
            if (self.dataset_path / 'labels' / split).exists():
                self.analyze_split(split)
                class_stats, imbalance_ratio = self.print_split_analysis(split)
                all_stats[split] = {
                    'class_stats': class_stats,
                    'imbalance': imbalance_ratio
                }
        
        # Overall summary
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_images = sum(self.image_counts.values())
        total_objects = sum(sum(counts.values()) for counts in self.class_counts.values())
        
        print(f"\nAcross All Splits:")
        print(f"  Total images: {total_images}")
        print(f"  Total objects: {total_objects}")
        print(f"  Number of classes: {len(self.class_names)}")
        
        # Aggregate class distribution
        aggregate_counts = defaultdict(int)
        for split_counts in self.class_counts.values():
            for class_id, count in split_counts.items():
                aggregate_counts[class_id] += count
        
        print(f"\nAggregate Class Distribution:")
        for class_id in sorted(self.class_names.keys()):
            count = aggregate_counts.get(class_id, 0)
            pct = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {self.class_names[class_id]:<25} {count:>7} ({pct:>6.2f}%)")
        
        overall_imbalance = self.calculate_imbalance_ratio(aggregate_counts)
        print(f"\nOverall Imbalance Ratio: {overall_imbalance:.2f}")
        
        # Print recommendations
        self.print_recommendations(all_stats)
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze class balance in YOLO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_dataset_balance.py data/animals_yolo
  python analyze_dataset_balance.py data/road_obstacles_final
        """
    )
    
    parser.add_argument('dataset', help='Path to YOLO dataset directory')
    
    args = parser.parse_args()
    
    try:
        analyzer = DatasetBalanceAnalyzer(args.dataset)
        analyzer.analyze()
        return 0
    except Exception as e:
        print(f"\n❌ ANALYSIS ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
