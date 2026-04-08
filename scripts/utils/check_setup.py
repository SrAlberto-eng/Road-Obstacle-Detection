"""
Script de verificación de configuración para Road Obstacle Detection.
Valida hardware, dataset y dependencias antes del entrenamiento.
"""

import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import sys

def check_gpu():
    """
    Verificar disponibilidad de GPU y configurar modo de entrenamiento.
    
    Returns:
        dict: Información de GPU y dispositivo recomendado
    """
    print("=" * 70)
    print("1. HARDWARE")
    print("=" * 70)
    
    print(f"PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    
    result = {
        'available': cuda_available,
        'device': '0' if cuda_available else 'cpu',
        'mode': 'GPU' if cuda_available else 'CPU'
    }
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        
        print(f"CUDA: {cuda_version}")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.2f} GB")
        print(f"Status: READY")
        print(f"Mode: GPU Acceleration")
        
        result['name'] = gpu_name
        result['vram'] = vram_gb
        result['cuda'] = cuda_version
    else:
        print(f"CUDA: Not available")
        print(f"GPU: Not detected")
        print(f"Status: CPU MODE")
        print(f"Mode: CPU only (training will be slower)")
        print(f"\nRecommendation: Use GPU-enabled machine for production training")
    
    return result

def check_dataset():
    """
    Verificar estructura y completitud del dataset.
    
    Returns:
        dict: Información del dataset y estado de verificación
    """
    print("\n" + "=" * 70)
    print("2. DATASET")
    print("=" * 70)
    
    data_yaml = Path('data/road_obstacles_final/data.yaml')
    
    if not data_yaml.exists():
        print(f"File: {data_yaml}")
        print(f"Status: NOT FOUND")
        return {'valid': False, 'error': 'data.yaml not found'}
    
    print(f"File: {data_yaml}")
    print(f"Status: Found")
    
    try:
        with open(data_yaml, 'r') as f:
            data_info = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Status: YAML PARSE ERROR")
        return {'valid': False, 'error': f'Invalid YAML: {e}'}
    
    num_classes = data_info.get('nc', 0)
    class_names = data_info.get('names', {})
    
    print(f"Classes: {num_classes}")
    print(f"Names: {list(class_names.values())}")
    
    base_path = Path(data_info['path'])
    splits = ['train', 'val', 'test']
    
    print(f"\nSplit verification:")
    split_stats = {}
    all_valid = True
    
    for split in splits:
        img_path = base_path / 'images' / split
        label_path = base_path / 'labels' / split
        
        if img_path.exists() and label_path.exists():
            num_images = len(list(img_path.glob('*.*')))
            num_labels = len(list(label_path.glob('*.txt')))
            
            # Tolerar pequeñas diferencias (< 1% o 100 imágenes)
            diff = abs(num_images - num_labels)
            diff_pct = (diff / num_images * 100) if num_images > 0 else 0
            
            status = "OK"
            valid = True
            if num_images == 0:
                status = "EMPTY"
                valid = False
                all_valid = False
            elif diff > 100 or diff_pct > 1.0:
                status = f"MISMATCH (img:{num_images} != lbl:{num_labels})"
                valid = False
                all_valid = False
            elif diff > 0:
                status = f"OK (minor diff: {diff})"
            
            print(f"  {split:>6}: {num_images:>5} images, {num_labels:>5} labels - {status}")
            split_stats[split] = {'images': num_images, 'labels': num_labels, 'valid': valid}
        else:
            print(f"  {split:>6}: DIRECTORY NOT FOUND")
            all_valid = False
            split_stats[split] = {'valid': False}
    
    total_images = sum(s.get('images', 0) for s in split_stats.values())
    print(f"\nTotal images: {total_images:,}")
    print(f"Dataset status: {'VALID' if all_valid else 'INVALID'}")
    
    return {
        'valid': all_valid,
        'num_classes': num_classes,
        'class_names': class_names,
        'total_images': total_images,
        'splits': split_stats
    }

def check_dependencies():
    """
    Verificar instalación de dependencias críticas.
    
    Returns:
        dict: Estado de dependencias
    """
    print("\n" + "=" * 70)
    print("3. DEPENDENCIES")
    print("=" * 70)
    
    dependencies = {}
    
    try:
        from ultralytics import __version__ as ultralytics_version
        print(f"Ultralytics: {ultralytics_version}")
        dependencies['ultralytics'] = ultralytics_version
    except ImportError:
        print(f"Ultralytics: NOT INSTALLED")
        dependencies['ultralytics'] = None
        return {'valid': False, 'dependencies': dependencies}
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
        dependencies['opencv'] = cv2.__version__
    except ImportError:
        print(f"OpenCV: NOT INSTALLED")
        dependencies['opencv'] = None
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        dependencies['numpy'] = np.__version__
    except ImportError:
        print(f"NumPy: NOT INSTALLED")
        dependencies['numpy'] = None
    
    try:
        import PIL
        print(f"Pillow: {PIL.__version__}")
        dependencies['pillow'] = PIL.__version__
    except ImportError:
        print(f"Pillow: NOT INSTALLED")
        dependencies['pillow'] = None
    
    all_valid = all(v is not None for v in dependencies.values())
    print(f"\nStatus: {'ALL DEPENDENCIES OK' if all_valid else 'MISSING DEPENDENCIES'}")
    
    return {'valid': all_valid, 'dependencies': dependencies}

def check_disk_space():
    """
    Verificar espacio disponible en disco.
    
    Returns:
        dict: Información de espacio en disco
    """
    print("\n" + "=" * 70)
    print("4. DISK SPACE")
    print("=" * 70)
    
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    print(f"Total: {total_gb:.2f} GB")
    print(f"Used: {used_gb:.2f} GB")
    print(f"Free: {free_gb:.2f} GB")
    
    sufficient = free_gb >= 5
    if sufficient:
        status = "SUFFICIENT"
    elif free_gb >= 2:
        status = "LOW (but usable)"
    else:
        status = "CRITICAL (< 2GB)"
    
    print(f"Status: {status}")
    
    if not sufficient:
        print(f"Recommendation: Free up disk space before training")
    
    return {
        'valid': free_gb >= 2,
        'free_gb': free_gb,
        'sufficient': sufficient
    }

def main():
    """Ejecutar todas las verificaciones del sistema."""
    print("\n" + "=" * 70)
    print("SYSTEM VERIFICATION")
    print("Road Obstacle Detection - Pre-training Check")
    print("=" * 70 + "\n")
    
    results = {}
    
    # 1. Hardware
    gpu_result = check_gpu()
    results['hardware'] = gpu_result
    
    # 2. Dataset
    dataset_result = check_dataset()
    results['dataset'] = dataset_result
    
    # 3. Dependencies
    deps_result = check_dependencies()
    results['dependencies'] = deps_result
    
    # 4. Disk Space
    disk_result = check_disk_space()
    results['disk'] = disk_result
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Hardware", gpu_result.get('available', False), 
         f"Mode: {gpu_result.get('mode', 'Unknown')}", 'warn'),  # CPU mode is warning, not fail
        ("Dataset", dataset_result.get('valid', False),
         f"{dataset_result.get('total_images', 0):,} images" if dataset_result.get('valid') else "Invalid structure", 'critical'),
        ("Dependencies", deps_result.get('valid', False),
         "All required packages installed" if deps_result.get('valid') else "Missing packages", 'critical'),
        ("Disk Space", disk_result.get('valid', False),
         f"{disk_result.get('free_gb', 0):.1f} GB free", 'critical')
    ]
    
    for name, valid, info, severity in checks:
        if valid:
            status = "[PASS]"
        elif severity == 'warn':
            status = "[WARN]"
        else:
            status = "[FAIL]"
        print(f"{status:>8} {name:<15} - {info}")
    
    # Only critical failures block training
    critical_checks = [(name, valid) for name, valid, _, severity in checks if severity == 'critical']
    all_valid = all(valid for _, valid in critical_checks)
    
    # Training configuration
    print("\n" + "=" * 70)
    if all_valid:
        print("STATUS: READY FOR TRAINING")
        print("\nRecommended training command:")
        device = gpu_result.get('device', 'cpu')
        
        if device == 'cpu':
            print(f"  python scripts/training/train.py --device cpu --batch 4 --epochs 50")
            print("\nNote: CPU training is significantly slower than GPU")
            print("      Estimated time: 10-20x slower than GPU training")
            print("      Consider using a GPU-enabled machine for production")
        else:
            vram = gpu_result.get('vram', 0)
            if vram < 6:
                batch_size = 4
            elif vram < 8:
                batch_size = 8
            else:
                batch_size = 16
            
            print(f"  python scripts/training/train.py --device {device} --batch {batch_size}")
            print(f"\nRecommended batch size for {vram:.1f}GB VRAM: {batch_size}")
    else:
        print("STATUS: NOT READY")
        print("\nCritical issues:")
        for name, valid, _, severity in checks:
            if not valid and severity == 'critical':
                print(f"  - Fix {name} configuration")
        print("\nRe-run this script after resolving issues")
    
    print("=" * 70)
    
    return all_valid

if __name__ == '__main__':
    main()
