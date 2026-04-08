"""
Script para generar evidencia de funcionamiento básico del sistema.
Realiza inferencia en imágenes de test y compila resultados para reporte académico.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import json
from datetime import datetime
import shutil

def test_basic_functionality(model_path, test_images_dir, output_dir):
    """
    Prueba el funcionamiento básico del modelo en imágenes de test.
    
    Args:
        model_path: Path al modelo entrenado (best.pt)
        test_images_dir: Directorio con imágenes de test
        output_dir: Directorio para guardar evidencia
    """
    print("="*70)
    print("PRUEBA DE FUNCIONAMIENTO BÁSICO DEL SISTEMA")
    print("="*70)
    
    model_path = Path(model_path)
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Verificar hardware
    print("\n1. VERIFICACIÓN DE HARDWARE")
    print("-"*70)
    print(f"PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("Modo: CPU")
        device = 'cpu'
    
    # 2. Cargar modelo
    print("\n2. CARGA DEL MODELO")
    print("-"*70)
    
    if not model_path.exists():
        print(f"ERROR: No se encontró el modelo en {model_path}")
        return
    
    print(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)
    
    print(f"Clases del modelo ({len(model.names)}):")
    for i, name in model.names.items():
        print(f"  {i}: {name}")
    
    # 3. Seleccionar imágenes de test
    print("\n3. SELECCIÓN DE IMÁGENES DE TEST")
    print("-"*70)
    
    if not test_images_dir.exists():
        print(f"ERROR: No se encontró {test_images_dir}")
        return
    
    # Obtener todas las imágenes de test
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(test_images_dir.glob(f'*{ext}')))
    
    print(f"Imágenes de test encontradas: {len(all_images)}")
    
    # Seleccionar 10 imágenes aleatorias para evidencia
    import random
    random.seed(42)
    num_samples = min(10, len(all_images))
    selected_images = random.sample(all_images, num_samples)
    
    print(f"Imágenes seleccionadas para prueba: {num_samples}")
    
    # 4. Ejecutar inferencia
    print("\n4. EJECUCIÓN DE INFERENCIA")
    print("-"*70)
    
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    results_data = []
    class_counts = {}
    total_detections = 0
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\nProcesando {i}/{num_samples}: {img_path.name}")
        
        # Realizar predicción
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.7,
            imgsz=640,
            device=device,
            save=False,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        num_detections = len(boxes)
        total_detections += num_detections
        
        print(f"  Detecciones: {num_detections}")
        
        # Contar clases detectadas
        img_classes = {}
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            if class_name not in img_classes:
                img_classes[class_name] = []
            img_classes[class_name].append(conf)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Guardar imagen con detecciones
        annotated_img = result.plot()
        output_path = predictions_dir / f"test_{i:02d}_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        
        # Guardar datos de esta predicción
        results_data.append({
            'image': img_path.name,
            'detections': num_detections,
            'classes': {cls: len(confs) for cls, confs in img_classes.items()},
            'avg_confidence': {cls: sum(confs)/len(confs) for cls, confs in img_classes.items()},
            'output_file': output_path.name
        })
        
        print(f"  Guardado: {output_path.name}")
    
    # 5. Generar reporte de evidencia
    print("\n5. GENERACIÓN DE REPORTE DE EVIDENCIA")
    print("-"*70)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EVIDENCIA DE FUNCIONAMIENTO BÁSICO DEL SISTEMA")
    report_lines.append("Road Obstacle Detection")
    report_lines.append("="*80)
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Modelo: {model_path}")
    report_lines.append("")
    
    # Hardware
    report_lines.append("="*80)
    report_lines.append("1. CONFIGURACIÓN DEL SISTEMA")
    report_lines.append("="*80)
    report_lines.append(f"PyTorch: {torch.__version__}")
    report_lines.append(f"CUDA: {'Disponible' if cuda_available else 'No disponible'}")
    if cuda_available:
        report_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    report_lines.append(f"Device: {device}")
    report_lines.append("")
    
    # Modelo
    report_lines.append("="*80)
    report_lines.append("2. MODELO CARGADO")
    report_lines.append("="*80)
    report_lines.append(f"Modelo: {model_path.name}")
    report_lines.append(f"Clases: {len(model.names)}")
    report_lines.append("Nombres de clases:")
    for i, name in model.names.items():
        report_lines.append(f"  {i}: {name}")
    report_lines.append("")
    
    # Pruebas realizadas
    report_lines.append("="*80)
    report_lines.append("3. PRUEBAS REALIZADAS")
    report_lines.append("="*80)
    report_lines.append(f"Imágenes de test procesadas: {num_samples}")
    report_lines.append(f"Total de detecciones: {total_detections}")
    report_lines.append(f"Promedio de detecciones por imagen: {total_detections/num_samples:.2f}")
    report_lines.append("")
    
    report_lines.append("Distribución de clases detectadas:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_detections * 100) if total_detections > 0 else 0
        report_lines.append(f"  {class_name}: {count} ({pct:.1f}%)")
    report_lines.append("")
    
    # Resultados por imagen
    report_lines.append("="*80)
    report_lines.append("4. RESULTADOS DETALLADOS POR IMAGEN")
    report_lines.append("="*80)
    report_lines.append("")
    
    for i, data in enumerate(results_data, 1):
        report_lines.append(f"Imagen {i}: {data['image']}")
        report_lines.append(f"  Total detecciones: {data['detections']}")
        if data['classes']:
            report_lines.append("  Clases detectadas:")
            for cls, count in data['classes'].items():
                avg_conf = data['avg_confidence'][cls]
                report_lines.append(f"    - {cls}: {count} (confianza promedio: {avg_conf:.3f})")
        else:
            report_lines.append("  No se detectaron objetos")
        report_lines.append(f"  Imagen con detecciones: {data['output_file']}")
        report_lines.append("")
    
    # Conclusiones
    report_lines.append("="*80)
    report_lines.append("5. CONCLUSIONES DEL FUNCIONAMIENTO BÁSICO")
    report_lines.append("="*80)
    report_lines.append("")
    
    if total_detections > 0:
        report_lines.append("[OK] El modelo carga correctamente")
        report_lines.append("[OK] El sistema realiza inferencia en imágenes nuevas")
        report_lines.append("[OK] Detecta objetos de las clases entrenadas")
        report_lines.append(f"[OK] Promedio de {total_detections/num_samples:.1f} detecciones por imagen")
        report_lines.append("")
        report_lines.append("Estado: FUNCIONAMIENTO BÁSICO VERIFICADO")
    else:
        report_lines.append("[!] No se detectaron objetos en las imágenes de prueba")
        report_lines.append("[!] Considerar ajustar umbral de confianza (--conf)")
        report_lines.append("")
        report_lines.append("Estado: REQUIERE REVISIÓN")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("FIN DEL REPORTE")
    report_lines.append("="*80)
    
    # Guardar reporte
    report_file = output_dir / 'EVIDENCIA_FUNCIONAMIENTO_BASICO.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReporte guardado: {report_file}")
    
    # 6. Generar resumen JSON
    summary = {
        'fecha': datetime.now().isoformat(),
        'modelo': str(model_path),
        'configuracion': {
            'device': str(device),
            'conf_threshold': 0.25,
            'iou_threshold': 0.7,
            'img_size': 640
        },
        'resultados': {
            'imagenes_procesadas': num_samples,
            'total_detecciones': total_detections,
            'promedio_detecciones': total_detections / num_samples if num_samples > 0 else 0,
            'clases_detectadas': class_counts
        },
        'imagenes_generadas': [str(p.relative_to(output_dir)) for p in predictions_dir.glob('*.jpg')]
    }
    
    json_file = output_dir / 'resumen_funcionamiento.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Resumen JSON guardado: {json_file}")
    
    # 7. Crear collage de 4 predicciones para reporte
    print("\n6. GENERANDO COLLAGE DE EVIDENCIA")
    print("-"*70)
    
    # Seleccionar 4 imágenes con más detecciones
    sorted_results = sorted(results_data, key=lambda x: x['detections'], reverse=True)
    top_4 = sorted_results[:4]
    
    collage_images = []
    for data in top_4:
        img_path = predictions_dir / data['output_file']
        img = cv2.imread(str(img_path))
        if img is not None:
            # Redimensionar para collage
            img = cv2.resize(img, (640, 640))
            collage_images.append(img)
    
    if len(collage_images) == 4:
        # Crear collage 2x2
        top_row = cv2.hconcat([collage_images[0], collage_images[1]])
        bottom_row = cv2.hconcat([collage_images[2], collage_images[3]])
        collage = cv2.vconcat([top_row, bottom_row])
        
        collage_path = output_dir / 'collage_funcionamiento_basico.jpg'
        cv2.imwrite(str(collage_path), collage)
        print(f"Collage guardado: {collage_path}")
    
    # Resumen final
    print("\n" + "="*70)
    print("GENERACIÓN DE EVIDENCIA COMPLETADA")
    print("="*70)
    print(f"\nArchivos generados en: {output_dir}")
    print(f"  1. EVIDENCIA_FUNCIONAMIENTO_BASICO.txt - Reporte completo")
    print(f"  2. resumen_funcionamiento.json - Datos JSON")
    print(f"  3. collage_funcionamiento_basico.jpg - Collage 2x2")
    print(f"  4. predictions/ - {num_samples} imágenes con detecciones")
    print("\nEstadísticas:")
    print(f"  - Imágenes procesadas: {num_samples}")
    print(f"  - Total detecciones: {total_detections}")
    print(f"  - Promedio por imagen: {total_detections/num_samples:.2f}")
    print("="*70)

def main():
    """Función principal"""
    print("\nGenerando evidencia de funcionamiento básico...")
    
    model_path = Path('results/training/road_obstacles_v1/weights/best.pt')
    test_images_dir = Path('data/road_obstacles_final/images/test')
    output_dir = Path('docs/evidencias/funcionamiento_basico')
    
    if not model_path.exists():
        print(f"\nERROR: No se encontró el modelo en {model_path}")
        print("Entrena el modelo primero con: python scripts/training/train.py")
        return
    
    if not test_images_dir.exists():
        print(f"\nERROR: No se encontró el directorio de test: {test_images_dir}")
        return
    
    test_basic_functionality(model_path, test_images_dir, output_dir)

if __name__ == '__main__':
    main()
