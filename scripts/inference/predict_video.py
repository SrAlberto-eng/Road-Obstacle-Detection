"""
Script para ejecutar detección de obstáculos en video
Usa el mejor modelo entrenado disponible
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def predict_video(model_path, video_path, conf=0.60, save=True):
    """Ejecuta detección en video"""
    
    print("=" * 70)
    print("DETECCIÓN DE OBSTÁCULOS EN VIDEO")
    print("=" * 70)
    print()
    
    # Verificar archivos
    model_path = Path(model_path)
    video_path = Path(video_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    print(f"Modelo: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confianza mínima: {conf}")
    print()
    
    # Cargar modelo
    print("Cargando modelo...")
    model = YOLO(str(model_path))
    print("✓ Modelo cargado")
    print()
    
    # Información del video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Información del video:")
    print(f"  Resolución: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames totales: {total_frames}")
    print(f"  Duración: {total_frames/fps:.1f} segundos")
    print()
    
    # Ejecutar predicción
    print("Ejecutando detección...")
    print()
    
    # YOLO puede procesar video directamente
    results = model.predict(
        source=str(video_path),
        conf=conf,
        save=save,
        stream=True,  # Stream para procesar frame por frame
        show=False,   # No mostrar ventana (OpenCV no soportado)
        verbose=True  # Mostrar progreso en consola
    )
    
    # Procesar resultados frame por frame
    frame_count = 0
    for result in results:
        frame_count += 1
        
        # Mostrar progreso cada 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            detections = len(result.boxes)
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - {detections} detecciones")
    
    print()
    print("=" * 70)
    print("DETECCIÓN COMPLETADA")
    print("=" * 70)
    print()
    
    if save:
        # YOLO guarda automáticamente en runs/detect/predict/
        output_path = Path("runs/detect/predict")
        print(f"Video con detecciones guardado en: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Detección de obstáculos en video")
    parser.add_argument(
        "--model",
        type=str,
        default="results/training/road_obstacles_binary_v1/weights/best.pt",
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/test.mp4",
        help="Ruta al video"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confianza mínima para detecciones (0-1)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar video con detecciones"
    )
    
    args = parser.parse_args()
    
    try:
        predict_video(
            model_path=args.model,
            video_path=args.video,
            conf=args.conf,
            save=not args.no_save
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

if __name__ == "__main__":
    main()
