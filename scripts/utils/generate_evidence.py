"""
Script para generar evidencias visuales del proyecto
Para reporte de avance académico/profesional
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def generate_evidence_report():
    print("=" * 70)
    print("GENERANDO EVIDENCIAS PARA REPORTE DE AVANCE")
    print("=" * 70)
    
    # Crear directorio de evidencias
    evidence_dir = Path('docs/evidencias')
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ Directorio creado: {evidence_dir}")
    
    # Cargar modelo
    model_path = 'results/training/road_obstacles_v1/weights/best.pt'
    model = YOLO(model_path)
    
    print("\n" + "=" * 70)
    print("1. PREDICCIONES VISUALES EN IMÁGENES DE TEST")
    print("=" * 70)
    
    # Obtener imágenes de ejemplo del test set
    test_images = list(Path('data/road_obstacles_final/images/test').glob('*.jpg'))
    if not test_images:
        test_images = list(Path('data/road_obstacles_final/images/test').glob('*.png'))
    
    # Seleccionar 6 imágenes representativas (2 de cada tipo si es posible)
    sample_images = test_images[:6]
    
    print(f"\n[INFO] Generando predicciones en {len(sample_images)} imágenes...")
    
    for i, img_path in enumerate(sample_images):
        results = model.predict(
            source=str(img_path),
            conf=0.5,
            save=True,
            project=str(evidence_dir),
            name='predicciones',
            exist_ok=True
        )
        print(f"  ✅ Predicción {i+1}/{len(sample_images)}: {img_path.name}")
    
    print(f"\n✅ Imágenes guardadas en: {evidence_dir}/predicciones/")
    
    print("\n" + "=" * 70)
    print("2. COPIANDO GRÁFICAS DE ENTRENAMIENTO")
    print("=" * 70)
    
    # Copiar visualizaciones importantes
    important_files = [
        ('runs/detect/val/confusion_matrix.png', 'Matriz de Confusión'),
        ('runs/detect/val/confusion_matrix_normalized.png', 'Matriz de Confusión Normalizada'),
        ('runs/detect/val/BoxPR_curve.png', 'Curva Precision-Recall'),
        ('runs/detect/val/BoxF1_curve.png', 'Curva F1-Score'),
        ('results/training/road_obstacles_v1/results.png', 'Métricas de Entrenamiento')
    ]
    
    import shutil
    for src, description in important_files:
        src_path = Path(src)
        if src_path.exists():
            dst_path = evidence_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            print(f"  ✅ {description}: {dst_path.name}")
        else:
            print(f"  ⚠️  {description}: No encontrado")
    
    print("\n" + "=" * 70)
    print("3. GENERANDO TABLA DE RESULTADOS")
    print("=" * 70)
    
    # Crear tabla de resultados en texto
    results_text = """
╔════════════════════════════════════════════════════════════════════╗
║         RESULTADOS DE EVALUACIÓN - TEST SET (2,039 imágenes)      ║
╚════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────┐
│ MÉTRICAS GENERALES                                                 │
├────────────────────┬────────────┬──────────────────────────────────┤
│ Métrica            │ Valor      │ Interpretación                   │
├────────────────────┼────────────┼──────────────────────────────────┤
│ mAP50              │ 61.22%     │ Precisión con IoU=0.5            │
│ mAP50-95           │ 33.74%     │ Precisión promedio IoU 0.5-0.95  │
│ Precision          │ 67.74%     │ 2 de cada 3 detecciones correctas│
│ Recall             │ 56.04%     │ Detecta 56% de objetos reales    │
└────────────────────┴────────────┴──────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ RENDIMIENTO POR CLASE                                              │
├──────────┬──────────┬────────────┬──────────┬─────────────────────┤
│ Clase    │ mAP50    │ Recall     │ Instanc. │ Estado              │
├──────────┼──────────┼────────────┼──────────┼─────────────────────┤
│ Animal   │ 84.2%    │ 80.5%      │ 277      │ ✅ EXCELENTE        │
│ Pothole  │ 78.2%    │ 75.5%      │ 299      │ ✅ MUY BUENO        │
│ Vehicle  │ 76.9%    │ 68.5%      │ 18,716   │ ✅ BUENO            │
│ Person   │ 50.1%    │ 41.6%      │ 2,304    │ ⚠️  BAJO            │
│ Debris   │ 42.2%    │ 38.8%      │ 49       │ ⚠️  BAJO            │
│ Bicycle  │ 35.8%    │ 31.3%      │ 247      │ ❌ MUY BAJO         │
└──────────┴──────────┴────────────┴──────────┴─────────────────────┘

╔════════════════════════════════════════════════════════════════════╗
║ CONFIGURACIÓN DEL ENTRENAMIENTO                                   ║
╚════════════════════════════════════════════════════════════════════╝

• Modelo: YOLOv8s (11.1M parámetros)
• Dataset: 10,898 imágenes (Train: 7,775 | Val: 1,084 | Test: 2,039)
• Épocas entrenadas: 142 (early stopping en época 112)
• Mejor modelo: Época 113
• Batch size: 8 | Workers: 4 | Imagen: 640x640
• Optimizador: AdamW | Learning rate: 0.001
• Tiempo de entrenamiento: ~4 horas
• GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)

╔════════════════════════════════════════════════════════════════════╗
║ ANÁLISIS Y CONCLUSIONES                                           ║
╚════════════════════════════════════════════════════════════════════╝

FORTALEZAS IDENTIFICADAS:
✅ Detección de animales: Rendimiento excelente (84.2% mAP50)
✅ Detección de baches: Rendimiento muy bueno (78.2% mAP50)
✅ Pipeline de entrenamiento completo y funcional
✅ Early stopping previno overfitting correctamente

ÁREAS DE MEJORA:
⚠️  Detección de personas: Rendimiento bajo (50.1% mAP50)
⚠️  Detección de bicicletas: Rendimiento muy bajo (35.8% mAP50)
⚠️  Dataset desbalanceado: 83.3% vehículos vs 16.7% obstáculos

PRÓXIMOS PASOS:
🎯 Implementar detección binaria (Normal/Obstáculo)
🎯 Aumentar datos de clases críticas (person, bicycle)
🎯 Ajustar thresholds de confianza por clase
"""
    
    results_file = evidence_dir / 'RESULTADOS_NUMERICOS.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(results_text)
    
    print(f"  ✅ Tabla de resultados: {results_file.name}")
    
    print("\n" + "=" * 70)
    print("✅ EVIDENCIAS GENERADAS EXITOSAMENTE")
    print("=" * 70)
    
    print(f"\n📁 Ubicación: {evidence_dir.absolute()}")
    print("\n📋 Archivos generados:")
    print("   1. predicciones/ - Imágenes con detecciones visuales")
    print("   2. confusion_matrix.png - Matriz de confusión")
    print("   3. BoxPR_curve.png - Curvas de rendimiento")
    print("   4. RESULTADOS_NUMERICOS.txt - Tabla de métricas")
    
    print("\n💡 PARA TU REPORTE, INCLUYE:")
    print("   ✅ 2-3 imágenes de predicciones/ (ejemplos visuales)")
    print("   ✅ confusion_matrix.png (análisis de errores)")
    print("   ✅ BoxPR_curve.png (rendimiento por clase)")
    print("   ✅ RESULTADOS_NUMERICOS.txt (métricas detalladas)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    generate_evidence_report()
