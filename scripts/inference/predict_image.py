"""
Predicción de obstáculos en imágenes con modelo dual.

Usa dos modelos:
- debris_pothole_v1: detecta escombros y baches
- yolov8s (COCO): detecta personas y animales

Genera imágenes con bounding boxes para presentaciones.

Uso:
    python predict_image.py                           # Usa imagen por defecto
    python predict_image.py --source data/test.jpg   # Imagen específica
    python predict_image.py --source data/imagenes/  # Carpeta de imágenes
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch
from datetime import datetime

# Clases de COCO que consideramos obstáculos (personas y animales)
COCO_OBSTACLE_CLASSES = {
    0: 'person', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
}

# Traducción al español
NOMBRES_ES = {
    'person': 'persona', 'bird': 'pajaro', 'cat': 'gato', 'dog': 'perro',
    'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante',
    'bear': 'oso', 'zebra': 'cebra', 'giraffe': 'jirafa',
    'debris': 'escombro', 'pothole': 'bache'
}


def dibujar_deteccion(frame, x1, y1, x2, y2, label, conf, color, grosor=3):
    """Dibuja un bounding box con etiqueta en el frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
    
    texto = f'{label} {conf:.0%}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.7
    grosor_texto = 2
    
    (tw, th), _ = cv2.getTextSize(texto, font, escala, grosor_texto)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, texto, (x1 + 5, y1 - 5), font, escala, (255, 255, 255), grosor_texto)


def dibujar_banner(frame, texto, color=(0, 0, 255)):
    """Dibuja un banner de alerta en la parte superior."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(texto, font, 1.3, 3)
    x = (w - tw) // 2
    cv2.putText(frame, texto, (x, 50), font, 1.3, (255, 255, 255), 3)
    return frame


def procesar_imagen(img_path, model_debris, model_coco, conf_debris=0.5, conf_coco=0.5):
    """Procesa una imagen con ambos modelos y retorna la imagen anotada."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  Error: No se pudo leer {img_path}")
        return None, {}
    
    h, w = frame.shape[:2]
    detecciones = {}
    
    # Modelo 1: debris/pothole
    results_debris = model_debris.predict(frame, conf=conf_debris, verbose=False)[0]
    if len(results_debris.boxes) > 0:
        for box in results_debris.boxes.cpu().numpy():
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            nombre = model_debris.names[cls]
            label = NOMBRES_ES.get(nombre, nombre)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            dibujar_deteccion(frame, x1, y1, x2, y2, label, conf, (0, 0, 255), 3)
            detecciones[label] = detecciones.get(label, 0) + 1
    
    # Modelo 2: COCO (personas/animales)
    results_coco = model_coco.predict(frame, conf=conf_coco, verbose=False)[0]
    if len(results_coco.boxes) > 0:
        for box in results_coco.boxes.cpu().numpy():
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            nombre = model_coco.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls in COCO_OBSTACLE_CLASSES:
                # Obstáculo (persona/animal) - rojo
                label = NOMBRES_ES.get(nombre, nombre)
                dibujar_deteccion(frame, x1, y1, x2, y2, label, conf, (0, 0, 255), 3)
                detecciones[label] = detecciones.get(label, 0) + 1
            else:
                # Otros objetos - gris
                dibujar_deteccion(frame, x1, y1, x2, y2, nombre, conf, (120, 120, 120), 2)
    
    # Banner de alerta si hay obstáculos
    if detecciones:
        tipos = ', '.join([f"{n} {t}" for t, n in detecciones.items()])
        frame = dibujar_banner(frame, f"ALERTA: {tipos.upper()}")
    
    # Info en la parte inferior
    total = sum(detecciones.values())
    info = f"Obstaculos detectados: {total}"
    cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame, detecciones


def main():
    parser = argparse.ArgumentParser(
        description='Detección de obstáculos con modelo dual (debris_pothole + COCO)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python predict_image.py                              # Usa data/test.jpg
  python predict_image.py --source data/mi_imagen.jpg
  python predict_image.py --source data/imagenes/     # Procesa carpeta
  python predict_image.py --conf 0.3 --show           # Menor umbral, mostrar
        """
    )
    parser.add_argument('--source', type=str, default='data/test1.jpg',
                        help='Imagen o carpeta de imágenes (default: data/test1.jpg)')
    parser.add_argument('--model', type=str, 
                        default='results/training/debris_pothole_v1/weights/best.pt',
                        help='Modelo para debris/potholes')
    parser.add_argument('--model-coco', type=str, default='yolov8s.pt',
                        help='Modelo COCO para personas/animales')
    parser.add_argument('--conf', type=float, default=0.7,
                        help='Umbral de confianza debris/pothole (default: 0.5)')
    parser.add_argument('--conf-coco', type=float, default=0.7,
                        help='Umbral de confianza COCO (default: 0.5)')
    parser.add_argument('--output', type=str, default='results/predictions',
                        help='Carpeta de salida')
    parser.add_argument('--show', action='store_true',
                        help='Mostrar imagen en ventana')
    
    args = parser.parse_args()
    
    # Verificar archivos
    source = Path(args.source)
    if not source.exists():
        print(f"Error: No existe {source}")
        return 1
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: No existe el modelo {model_path}")
        return 1
    
    print("=" * 60)
    print("DETECCIÓN DE OBSTÁCULOS - MODELO DUAL")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Modelo debris/pothole: {args.model}")
    print(f"Modelo COCO: {args.model_coco}")
    print(f"Confianza: {args.conf:.0%} (debris) / {args.conf_coco:.0%} (COCO)")
    print()
    
    # Cargar modelos
    print("Cargando modelos...")
    model_debris = YOLO(str(args.model))
    model_coco = YOLO(args.model_coco)
    print(f"  Clases debris/pothole: {list(model_debris.names.values())}")
    print()
    
    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener lista de imágenes
    if source.is_file():
        imagenes = [source]
    else:
        imagenes = list(source.glob('*.jpg')) + list(source.glob('*.png')) + list(source.glob('*.jpeg'))
    
    if not imagenes:
        print(f"No se encontraron imágenes en {source}")
        return 1
    
    print(f"Procesando {len(imagenes)} imagen(es)...")
    print("-" * 60)
    
    total_detecciones = {}
    
    for img_path in imagenes:
        print(f"\n  {img_path.name}")
        
        # Procesar imagen
        resultado, detecciones = procesar_imagen(
            img_path, model_debris, model_coco, 
            args.conf, args.conf_coco
        )
        
        if resultado is None:
            continue
        
        # Acumular estadísticas
        for tipo, n in detecciones.items():
            total_detecciones[tipo] = total_detecciones.get(tipo, 0) + n
        
        # Guardar resultado
        timestamp = datetime.now().strftime("%H%M%S")
        nombre_salida = f"{img_path.stem}_detected_{timestamp}.jpg"
        ruta_salida = output_dir / nombre_salida
        cv2.imwrite(str(ruta_salida), resultado)
        print(f"    Guardado: {ruta_salida}")
        
        if detecciones:
            for tipo, n in detecciones.items():
                print(f"    - {n} {tipo}")
        else:
            print(f"    - Sin obstáculos detectados")
        
        # Mostrar si se solicita
        if args.show:
            cv2.imshow('Detecciones', resultado)
            print("    Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Resumen final
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes procesadas: {len(imagenes)}")
    print(f"Resultados guardados en: {output_dir}")
    
    if total_detecciones:
        print(f"\nDetecciones totales:")
        for tipo, n in sorted(total_detecciones.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tipo}: {n}")
    
    print("=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
