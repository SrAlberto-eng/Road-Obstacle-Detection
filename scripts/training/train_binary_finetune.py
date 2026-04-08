"""
Script de entrenamiento para detección binaria de obstáculos en carretera.

Este script entrena un modelo YOLOv8 usando fine-tuning (ajuste fino) para
clasificar objetos en dos categorías: normal y obstáculo.

Uso:
    python scripts/training/train_binary_finetune.py --datos data/road_obstacles_final/data.yaml --epocas 100
"""

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path
import yaml

def procesar_argumentos():
    """
    Procesa los argumentos de línea de comandos.
    
    Returns:
        Namespace: Argumentos parseados con la configuración del entrenamiento
    """
    parser = argparse.ArgumentParser(
        description='Entrenamiento de YOLOv8 para detección binaria de obstáculos'
    )
    
    parser.add_argument(
        '--datos', 
        type=str, 
        default='data/road_obstacles_final/data.yaml',
        help='Ruta al archivo data.yaml del dataset'
    )
    
    parser.add_argument(
        '--modelo', 
        type=str, 
        default='yolov8s.pt',
        help='Modelo base pre-entrenado (yolov8n.pt, yolov8s.pt, yolov8m.pt)'
    )
    
    parser.add_argument(
        '--epocas', 
        type=int, 
        default=100,
        help='Número de épocas de entrenamiento'
    )
    
    parser.add_argument(
        '--batch', 
        type=int, 
        default=8,
        help='Tamaño del batch (ajustar según memoria GPU disponible)'
    )
    
    parser.add_argument(
        '--tamaño-img', 
        type=int, 
        default=640,
        help='Tamaño de la imagen para entrenamiento'
    )
    
    parser.add_argument(
        '--dispositivo', 
        type=str, 
        default='0',
        help='Dispositivo de entrenamiento (0 para GPU, cpu para CPU)'
    )
    
    parser.add_argument(
        '--proyecto', 
        type=str, 
        default='results/training',
        help='Carpeta donde guardar los resultados'
    )
    
    parser.add_argument(
        '--nombre', 
        type=str, 
        default='road_obstacles_binary',
        help='Nombre del experimento'
    )
    
    parser.add_argument(
        '--congelar', 
        type=int, 
        default=10,
        help='Número de capas a congelar (0 = ninguna, 10 = backbone completo)'
    )
    
    parser.add_argument(
        '--freeze', 
        type=int, 
        default=10,
        help='Número de capas a congelar (0 = ninguna, 10 = backbone completo)'
    )
    
    return parser.parse_args()


def imprimir_info_entrenamiento(args, info_datos):
    """
    Imprime la información de configuración del entrenamiento.
    
    Args:
        args: Argumentos parseados
        info_datos: Información del dataset cargada del yaml
    """
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE ENTRENAMIENTO - DETECCIÓN BINARIA DE OBSTÁCULOS")
    print("="*70)
    
    print("\nDATASET:")
    print(f"  Archivo: {args.datos}")
    print(f"  Clases: {info_datos.get('nc', 'N/A')}")
    print(f"  Nombres: {info_datos.get('names', 'N/A')}")
    
    print("\nMODELO:")
    print(f"  Base: {args.modelo}")
    print(f"  Capas congeladas: {args.congelar}")
    print(f"  Estrategia: Fine-tuning con transfer learning")
    
    print("\nHIPERPARÁMETROS:")
    print(f"  Épocas: {args.epocas}")
    print(f"  Batch size: {args.batch}")
    print(f"  Tamaño imagen: {args.tamaño_img}x{args.tamaño_img}")
    print(f"  Learning rate inicial: 0.01 (ajuste automático)")
    print(f"  Patience: 50 épocas (early stopping)")
    
    print("\nHARDWARE:")
    nombre_dispositivo = "GPU (CUDA)" if args.dispositivo == '0' and torch.cuda.is_available() else "CPU"
    print(f"  Dispositivo: {nombre_dispositivo}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\nSALIDA:")
    print(f"  Carpeta proyecto: {args.proyecto}")
    print(f"  Nombre experimento: {args.nombre}")
    print(f"  Modelo guardado: {args.proyecto}/{args.nombre}/weights/best.pt")
    
    print("\n" + "="*70)
    print()


def principal():
    """
    Función principal que ejecuta el entrenamiento del modelo.
    
    Proceso:
    1. Cargar configuración y validar archivos
    2. Inicializar modelo YOLOv8 pre-entrenado
    3. Configurar hiperparámetros
    4. Entrenar con early stopping
    5. Guardar mejor modelo
    """
    # Procesar argumentos de línea de comandos
    args = procesar_argumentos()
    
    # Validar que existe el archivo de datos
    ruta_datos = Path(args.datos)
    if not ruta_datos.exists():
        print(f"Error: No se encuentra el archivo {ruta_datos}")
        print("Asegúrate de que el dataset esté en la ubicación correcta.")
        return
    
    # Cargar información del dataset
    with open(ruta_datos, 'r') as f:
        info_datos = yaml.safe_load(f)
    
    # Mostrar información de configuración
    imprimir_info_entrenamiento(args, info_datos)
    
    # Cargar modelo pre-entrenado
    print("Cargando modelo YOLOv8 pre-entrenado...")
    modelo = YOLO(args.modelo)
    print(f"Modelo {args.modelo} cargado correctamente")
    print()
    
    # Configurar entrenamiento
    print("Iniciando entrenamiento...")
    print("(Este proceso puede tomar varias horas dependiendo del hardware)")
    print()
    
    try:
        # Entrenar el modelo
        resultados = modelo.train(
            # Dataset
            data=str(ruta_datos),
            
            # Hiperparámetros básicos
            epochs=args.epocas,
            batch=args.batch,
            imgsz=args.tamaño_img,
            device=args.dispositivo,
            
            # Optimización
            optimizer='AdamW',           # Optimizador Adam con weight decay
            lr0=0.01,                    # Learning rate inicial
            lrf=0.01,                    # Learning rate final (1% del inicial)
            momentum=0.937,              # Momentum para SGD
            weight_decay=0.0005,         # Regularización L2
            
            # Fine-tuning
            freeze=args.congelar,        # Congelar primeras N capas
            
            # Early stopping
            patience=50,                 # Detener si no mejora en 50 épocas
            
            # Data augmentation (aumento de datos)
            hsv_h=0.015,                # Variación de matiz (color)
            hsv_s=0.7,                  # Variación de saturación
            hsv_v=0.4,                  # Variación de valor (brillo)
            degrees=0.0,                # Rotación (0 = desactivado)
            translate=0.1,              # Traslación
            scale=0.5,                  # Escala
            shear=0.0,                  # Cizallamiento
            perspective=0.0,            # Perspectiva
            flipud=0.5,                 # Volteo vertical (importante para carreteras)
            fliplr=0.5,                 # Volteo horizontal
            mosaic=1.0,                 # Mosaic augmentation
            mixup=0.0,                  # Mixup augmentation
            copy_paste=0.0,             # Copy-paste augmentation
            
            # Configuración de salida
            project=args.proyecto,
            name=args.nombre,
            exist_ok=False,             # No sobrescribir experimentos previos
            pretrained=True,            # Usar pesos pre-entrenados
            verbose=True,               # Mostrar información detallada
            
            # Validación
            val=True,                   # Validar durante entrenamiento
            
            # Workers (0 para Windows para evitar problemas de multiprocessing)
            workers=0,
            
            # Cache (puede acelerar pero usa más RAM)
            cache=False,
        )
        
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"\nMejor modelo guardado en:")
        print(f"  {args.proyecto}/{args.nombre}/weights/best.pt")
        print(f"\nÚltimo modelo guardado en:")
        print(f"  {args.proyecto}/{args.nombre}/weights/last.pt")
        print(f"\nMétricas y gráficas guardadas en:")
        print(f"  {args.proyecto}/{args.nombre}/")
        print("\n" + "="*70)
        
        # Mostrar mejores métricas
        if resultados.results_dict:
            print("\nMEJORES MÉTRICAS OBTENIDAS:")
            metricas = resultados.results_dict
            if 'metrics/mAP50(B)' in metricas:
                print(f"  mAP50: {metricas['metrics/mAP50(B)']:.3f}")
            if 'metrics/precision(B)' in metricas:
                print(f"  Precision: {metricas['metrics/precision(B)']:.3f}")
            if 'metrics/recall(B)' in metricas:
                print(f"  Recall: {metricas['metrics/recall(B)']:.3f}")
            print()

        
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario")
        print("El progreso se ha guardado. Puedes reanudar el entrenamiento más tarde.")
        
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")
        print("Revisa la configuración y los logs para más detalles.")
        raise

if __name__ == "__main__":
    principal()
