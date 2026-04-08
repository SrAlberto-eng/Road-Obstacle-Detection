"""
Script de entrenamiento optimizado para Road Obstacle Detection
Dataset: 10,898 imágenes, 6 clases con imbalance severo (75.9x)
"""

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path
import yaml
from collections import defaultdict
import sys

def calculate_class_weights(data_yaml_path, vehicle_weight_reduction=0.1, obstacle_weight_boost=2.5):
    """
    Calcula class weights para manejar imbalance del dataset.
    
    Args:
        data_yaml_path: Path al archivo data.yaml
        vehicle_weight_reduction: Factor de reducción para vehículos (default: 0.1)
        obstacle_weight_boost: Factor de aumento para obstáculos (default: 2.5)
    
    Returns:
        tuple: (weights, class_counts, total_objects, class_names)
    """
    data_yaml_path = Path(data_yaml_path)
    dataset_root = data_yaml_path.parent
    
    with open(data_yaml_path, 'r') as f:
        data_info = yaml.safe_load(f)
    
    class_names = data_info.get('names', {})
    num_classes = len(class_names)
    
    train_labels_dir = dataset_root / 'labels' / 'train'
    class_counts = defaultdict(int)
    
    if train_labels_dir.exists():
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    
    total_objects = sum(class_counts.values())
    
    weights = {}
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_objects / (num_classes * count)
        weights[class_id] = weight
    
    vehicle_class_id = 0
    weights[vehicle_class_id] *= vehicle_weight_reduction
    
    obstacle_classes = [1, 2, 3, 4, 5]
    for class_id in obstacle_classes:
        if class_id in weights:
            weights[class_id] *= obstacle_weight_boost
    
    return weights, class_counts, total_objects, class_names


def print_training_info(data_yaml, data_info, weights, counts, total, class_names):
    """Imprime información del dataset y distribución de clases."""
    print("\n" + "="*70)
    print("DATASET")
    print("="*70)
    print(f"Ubicación: {data_yaml}")
    print(f"Clases: {data_info.get('nc', 'N/A')}")
    print(f"Nombres: {list(data_info.get('names', {}).values())}")
    
    print("\n" + "="*70)
    print("DISTRIBUCIÓN DE CLASES Y WEIGHTS")
    print("="*70)
    print(f"{'Clase':<12} {'Count':<10} {'%':<8} {'Weight':<12} {'Prioridad'}")
    print("-"*70)
    
    for class_id in sorted(class_names.keys()):
        name = class_names[class_id]
        count = counts.get(class_id, 0)
        pct = (count / total * 100) if total > 0 else 0
        weight = weights.get(class_id, 0)
        priority = "BAJA" if class_id == 0 else ("CRITICA" if class_id in [1, 3] else "ALTA")
        print(f"{name:<12} {count:<10,} {pct:<7.2f}% {weight:<12.2f} {priority}")
    
    max_count = max(counts.values()) if counts else 1
    min_count = min(counts.values()) if counts else 1
    imbalance = max_count / min_count if min_count > 0 else 0
    
    print("-"*70)
    print(f"Total objetos: {total:,}")
    print(f"Imbalance ratio: {imbalance:.1f}x")
    print(f"Ajustes: vehicle x0.1, obstacles x2.5")
    print("="*70)

def parse_args():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento de YOLOv8 para Road Obstacle Detection')
    parser.add_argument('--data', type=str, default='data/road_obstacles_final/data.yaml',
                        help='Path al archivo data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Modelo base (yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Número de épocas')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Tamaño de imagen')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0, 1, 2... o cpu)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Épocas de paciencia para early stopping')
    parser.add_argument('--project', type=str, default='results/training',
                        help='Directorio de resultados')
    parser.add_argument('--name', type=str, default='road_obstacles_v1',
                        help='Nombre del experimento')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Sobrescribir experimento existente')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        print("\n" + "="*70)
        print("CONFIGURACIÓN DEL SISTEMA")
        print("="*70)
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        print("="*70)
        
        data_yaml = Path(args.data)
        
        if not data_yaml.exists():
            print(f"\n[ERROR] No se encontró el archivo: {data_yaml}")
            print("Verifica que el dataset esté en la ubicación correcta.")
            sys.exit(1)
        
        with open(data_yaml, 'r') as f:
            data_info = yaml.safe_load(f)
        
        weights, counts, total, class_names = calculate_class_weights(
            data_yaml, 
            vehicle_weight_reduction=0.1, 
            obstacle_weight_boost=2.5
        )
        
        print_training_info(data_yaml, data_info, weights, counts, total, class_names)
        
        model = YOLO(args.model)
        
        print("\n" + "="*70)
        print("INICIANDO ENTRENAMIENTO")
        print("="*70)
        
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.001,
            weight_decay=0.0005,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.15,
            scale=0.9,
            shear=2.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            cos_lr=True,
            close_mosaic=10,
            amp=True,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            plots=True,
            save=True,
            save_period=10,
            val=True,
            verbose=True,
            seed=42
        )
        
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"\nResultados guardados en: {args.project}/{args.name}")
        print("\nPróximos pasos:")
        print(f"  1. Revisar métricas: {args.project}/{args.name}/results.png")
        print(f"  2. Validar modelo: python scripts/evaluation/evaluate_model.py --weights {args.project}/{args.name}/weights/best.pt")
        print(f"  3. Hacer predicciones: python scripts/inference/predict_image.py --weights {args.project}/{args.name}/weights/best.pt")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Archivo no encontrado: {e}")
        print("Verifica que el dataset y modelo estén en las rutas correctas.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"\n[ERROR] Error al leer el archivo YAML: {e}")
        print("Verifica que data.yaml tenga el formato correcto.")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] GPU sin memoria suficiente.")
            print("Soluciones:")
            print("  1. Reducir batch size: --batch 4")
            print("  2. Reducir tamaño de imagen: --imgsz 512")
            print("  3. Usar modelo más pequeño: --model yolo11n.pt")
        else:
            print(f"\n[ERROR] Error durante el entrenamiento: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Entrenamiento interrumpido por el usuario.")
        print("El progreso se ha guardado en los checkpoints.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
