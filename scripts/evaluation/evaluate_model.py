"""
Script para evaluar el modelo entrenado en el test set
Muestra metricas generales y por clase
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    print("=" * 70)
    print("EVALUACION DEL MODELO EN TEST SET")
    print("=" * 70)
    
    # Ruta al mejor modelo
    model_path = Path('results/training/road_obstacles_v1/weights/best.pt')
    
    if not model_path.exists():
        print(f"\n[ERROR] No se encontro el modelo: {model_path}")
        exit(1)
    
    print(f"\n[OK] Modelo encontrado: {model_path}")
    print(f"Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Cargar modelo
    print("\n" + "=" * 70)
    print("CARGANDO MODELO...")
    print("=" * 70)
    model = YOLO(str(model_path))
    
    # Evaluar en test set
    print("\n" + "=" * 70)
    print("EVALUANDO EN TEST SET (2,039 imagenes)...")
    print("=" * 70)
    print("\nEsto puede tomar varios minutos...\n")
    
    results = model.val(
        data='data/road_obstacles_final/data.yaml',
        split='test',  # Usar test set
        batch=8,       # Mismo batch que entrenamiento
        imgsz=640,
        verbose=True,
        plots=True,    # Generar graficas
        save_json=False
    )
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS EN TEST SET")
    print("=" * 70)
    
    print(f"\n{'Metrica':<20} {'Valor':<10}")
    print("-" * 35)
    print(f"{'mAP50':<20} {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
    print(f"{'mAP50-95':<20} {results.box.map:.4f} ({results.box.map*100:.2f}%)")
    print(f"{'Precision':<20} {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
    print(f"{'Recall':<20} {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
    
    # Metricas por clase
    print("\n" + "=" * 70)
    print("METRICAS POR CLASE")
    print("=" * 70)
    
    class_names = ['vehicle', 'person', 'bicycle', 'animal', 'pothole', 'debris']
    
    print(f"\n{'Clase':<12} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 65)
    
    for i, class_name in enumerate(class_names):
        if i < len(results.box.ap_class_index):
            p = results.box.p[i] if i < len(results.box.p) else 0
            r = results.box.r[i] if i < len(results.box.r) else 0
            ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
            ap = results.box.ap[i] if i < len(results.box.ap) else 0
            
            print(f"{class_name:<12} {p:<12.4f} {r:<12.4f} {ap50:<12.4f} {ap:<12.4f}")
    
    print("\n" + "=" * 70)
    print("ANALISIS DE CLASES CRITICAS (Obstaculos)")
    print("=" * 70)
    
    # Clases criticas: person, animal, pothole, debris, bicycle
    critical_classes = [1, 3, 4, 5, 2]  # indices
    critical_names = ['person', 'animal', 'pothole', 'debris', 'bicycle']
    
    print("\nClases de ALTA PRIORIDAD para deteccion de obstaculos:")
    for idx, name in zip(critical_classes, critical_names):
        if idx < len(results.box.ap50):
            ap50 = results.box.ap50[idx]
            r = results.box.r[idx] if idx < len(results.box.r) else 0
            print(f"  {name:<10} - mAP50: {ap50:.4f} ({ap50*100:.1f}%), Recall: {r:.4f} ({r*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("[OK] EVALUACION COMPLETADA")
    print("=" * 70)
    print(f"\nResultados guardados en: {model_path.parent.parent}")
    print("Graficas de confusion matrix y curvas disponibles en el directorio")
    print("=" * 70)

if __name__ == '__main__':
    main()
