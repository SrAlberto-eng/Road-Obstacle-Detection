"""
Script para diagnosticar problemas del modelo
- Detecta falsos positivos
- Muestra confianzas de predicciones
- Analiza detecciones en background
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def main():
    print("=" * 70)
    print("DIAGNOSTICO DEL MODELO")
    print("=" * 70)
    
    # Cargar modelo
    model_path = 'results/training/road_obstacles_v1/weights/best.pt'
    model = YOLO(model_path)
    
    print(f"\n[OK] Modelo cargado: {model_path}")
    
    # Probar con una imagen del test set
    test_images = list(Path('data/road_obstacles_final/images/test').glob('*.jpg'))[:5]
    
    if not test_images:
        test_images = list(Path('data/road_obstacles_final/images/test').glob('*.png'))[:5]
    
    print(f"\n[INFO] Probando con {len(test_images)} imagenes del test set")
    print("=" * 70)
    
    for img_path in test_images:
        print(f"\n--- Imagen: {img_path.name} ---")
        
        # Predecir con diferentes thresholds
        for conf_threshold in [0.25, 0.5, 0.7]:
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                verbose=False,
                save=False
            )
            
            detections = results[0].boxes
            print(f"\nConf >= {conf_threshold}: {len(detections)} detecciones")
            
            # Mostrar detalles de cada deteccion
            if len(detections) > 0:
                for i, box in enumerate(detections):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_names = ['vehicle', 'person', 'bicycle', 'animal', 'pothole', 'debris']
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    
                    print(f"  [{i+1}] {cls_name:10} - Confianza: {conf:.3f}")
    
    print("\n" + "=" * 70)
    print("RECOMENDACIONES:")
    print("=" * 70)
    print("\n1. Si ves muchas detecciones con confianza < 0.5:")
    print("   -> Usa conf=0.5 o mayor en inferencia")
    print("\n2. Si detecta background como objetos:")
    print("   -> Revisa las etiquetas originales del dataset")
    print("   -> Considera agregar clase 'background' o imagenes negativas")
    print("\n3. Si confunde clases:")
    print("   -> Revisa confusion matrix en:")
    print("   -> runs/detect/val/confusion_matrix.png")
    print("=" * 70)

if __name__ == '__main__':
    main()
