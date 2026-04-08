"""
Script de detección de obstáculos en carretera en tiempo real.

Este script procesa un video mostrando las detecciones del modelo YOLO en tiempo real.
Detecta:
- Debris y potholes (del modelo entrenado)
- Personas y animales (clases de COCO pre-entrenadas)

Todas las detecciones anteriores se marcan como obstáculos y generan alertas visuales.

Características:
- Visualización en tiempo real con pygame
- Alertas visuales cuando se detectan obstáculos
- Controles interactivos (pausa/reanudar)
- Opción de guardar el video procesado
- Información de progreso y estadísticas en pantalla
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import numpy as np
import time
import pygame
import sys

# Clases de COCO que consideramos obstáculos peligrosos
# ID de clases en COCO que corresponden a personas y animales
COCO_OBSTACLE_CLASSES = {
    0: 'person',      # Persona
    14: 'bird',       # Pájaro
    15: 'cat',        # Gato
    16: 'dog',        # Perro
    17: 'horse',      # Caballo
    18: 'sheep',      # Oveja
    19: 'cow',        # Vaca
    20: 'elephant',   # Elefante
    21: 'bear',       # Oso
    22: 'zebra',      # Cebra
    23: 'giraffe'     # Jirafa
}

# Traducción al español
COCO_NAMES_ES = {
    'person': 'persona',
    'bird': 'pajaro',
    'cat': 'gato',
    'dog': 'perro',
    'horse': 'caballo',
    'sheep': 'oveja',
    'cow': 'vaca',
    'elephant': 'elefante',
    'bear': 'oso',
    'zebra': 'cebra',
    'giraffe': 'jirafa'
}

def parse_args():
    """
    Procesa los argumentos de línea de comandos.
    
    Returns:
        Namespace: Argumentos parseados con la configuración del script
    """
    parser = argparse.ArgumentParser(description='Detección de obstáculos en video con alertas en tiempo real')
    parser.add_argument('--video', type=str, default='data/test.mp4',
                       help='Ruta al video de entrada')
    parser.add_argument('--model', type=str, 
                       default='results/training/debris_pothole_v1/weights/best.pt',
                       help='Ruta al modelo entrenado para debris/potholes')
    parser.add_argument('--model-coco', type=str, default='yolov8s.pt',
                       help='Ruta al modelo COCO para personas/animales (default: yolov8s.pt)')
    parser.add_argument('--conf', type=float, default=0.75,
                       help='Umbral de confianza para las detecciones (default: 0.75)')
    parser.add_argument('--conf-coco', type=float, default=0.5,
                       help='Umbral de confianza para detecciones COCO (default: 0.5)')
    parser.add_argument('--save', action='store_true',
                       help='Guardar el video procesado en la carpeta results/videos')
    parser.add_argument('--fps', type=int, default=None,
                       help='FPS para reproducción (None = velocidad original del video)')
    return parser.parse_args()

def draw_alert_banner(frame, text, color=(0, 0, 255)):
    """
    Dibuja un banner de alerta semi-transparente en la parte superior del frame.
    
    Args:
        frame: Frame de video (numpy array)
        text: Texto a mostrar en el banner
        color: Color del banner en formato BGR (default: rojo)
        
    Returns:
        Frame modificado con el banner de alerta
    """
    h, w = frame.shape[:2]
    
    # Crear overlay semi-transparente para el banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Configurar y dibujar el texto centrado
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 50
    cv2.putText(frame, text, (text_x, text_y), font, 1.5, (255, 255, 255), 3)
    
    return frame

def draw_detection(frame, x1, y1, x2, y2, label, conf, color, thickness=3):
    """
    Dibuja un bounding box con su etiqueta en el frame.
    
    Args:
        frame: Frame de video (numpy array)
        x1, y1, x2, y2: Coordenadas del bounding box
        label: Etiqueta de la clase en español
        conf: Confianza de la detección
        color: Color del bounding box en formato BGR
        thickness: Grosor del borde (default: 3)
    """
    # Dibujar el rectángulo del bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Crear y dibujar la etiqueta con clase y confianza
    label_text = f'{label} {conf:.2f}'
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main():
    """
    Función principal que ejecuta el flujo completo de detección en tiempo real.
    
    Proceso:
    1. Cargar y validar el modelo y video
    2. Inicializar la ventana de visualización con pygame
    3. Procesar frame por frame aplicando el modelo YOLO
    4. Dibujar detecciones y alertas en cada frame
    5. Mostrar en pantalla con controles interactivos
    """
    args = parse_args()
    
    # Verificar que los archivos necesarios existen
    video_path = Path(args.video)
    model_path = Path(args.model)
    
    if not video_path.exists():
        print(f"Error: No se encuentra el video {video_path}")
        return
    
    if not model_path.exists():
        print(f"Error: No se encuentra el modelo {model_path}")
        return
    
    # Verificar ruta del modelo COCO
    model_coco_path = Path(args.model_coco)
    
    print()
    print("Controles:")
    print("  ESPACIO - Pausar/Reanudar")
    print("  Q - Salir")
    print()
    
    # Cargar el modelo entrenado para debris/potholes
    model = YOLO(str(model_path))
    
    # Cargar el modelo COCO para personas/animales
    model_coco = YOLO(str(model_coco_path))
    
    # Abrir el video con OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se puede abrir el video {video_path}")
        return
    
    # Extraer propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    # Determinar FPS de reproducción (original o personalizado)
    playback_fps = args.fps if args.fps else fps
    
    # Mostrar información del video
    print(f"Informacion del video:")
    print(f"  Resolucion: {width}x{height}")
    print(f"  FPS original: {fps:.2f}")
    print(f"  FPS reproduccion: {playback_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duracion: {duration:.1f} segundos")
    print()
    
    # Configurar guardado de video procesado si se solicita
    out = None
    if args.save:
        output_dir = Path("results/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Guardando video procesado en: {output_path}")
        print()
    
    
    # Inicializar pygame para la visualización
    pygame.init()
    
    # Calcular tamaño de ventana (limitar a 1280px de ancho si es muy grande)
    display_width = min(width, 1280)
    display_height = int(height * (display_width / width))
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Deteccion de Obstaculos - Q: Salir, ESPACIO: Pausar')
    clock = pygame.time.Clock()
    
    # Variables de control del bucle principal
    frame_count = 0
    paused = False
    running = True
    
    try:
        while cap.isOpened() and running:
            # Procesar eventos de pygame (cierre de ventana, teclas)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            print("Pausado")
                        else:
                            print("Reanudado")
            
            if not paused:
                # Leer el siguiente frame del video
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # ========== MODELO 1: best.pt ==========
                results_debris = model.predict(frame, conf=args.conf, verbose=False)[0]
                
                # ========== MODELO 2: yolov8s.pt ==========
                results_coco = model_coco.predict(frame, conf=args.conf_coco, verbose=False)[0]
                
                # Inicializar contadores de detecciones por tipo
                debris_pothole_count = 0  # Debris y potholes del modelo
                person_animal_count = 0   # Personas y animales de COCO
                other_count = 0           # Otras clases de COCO (no obstáculos)
                
                # Diccionario para contar tipos específicos
                detection_types = {}
                
                # ========== PROCESAR DETECCIONES DEL MODELO DEBRIS/POTHOLE ==========
                if len(results_debris.boxes) > 0:
                    boxes = results_debris.boxes.cpu().numpy()
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Debris o pothole siempre son obstáculos
                        debris_pothole_count += 1
                        label_es = 'escombro' if class_name == 'debris' else 'bache'
                        detection_types[label_es] = detection_types.get(label_es, 0) + 1
                        
                        # Dibujar con color rojo (obstáculos)
                        draw_detection(frame, x1, y1, x2, y2, label_es, conf, 
                                     color=(0, 0, 255), thickness=3)
                
                # ========== PROCESAR DETECCIONES DEL MODELO COCO ==========
                if len(results_coco.boxes) > 0:
                    boxes = results_coco.boxes.cpu().numpy()
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model_coco.names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Determinar si es obstáculo (persona/animal) u otro objeto
                        if cls in COCO_OBSTACLE_CLASSES:
                            # Persona o animal → OBSTÁCULO
                            person_animal_count += 1
                            label_es = COCO_NAMES_ES.get(class_name, class_name)
                            detection_types[label_es] = detection_types.get(label_es, 0) + 1
                            
                            # Dibujar con color rojo (obstáculos)
                            draw_detection(frame, x1, y1, x2, y2, label_es, conf, 
                                         color=(0, 0, 255), thickness=3)
                        else:
                            # Otros objetos (vehículos, señales, etc.) → NO OBSTÁCULO
                            other_count += 1
                            label_es = class_name
                            
                            # Dibujar con color gris (objetos normales)
                            draw_detection(frame, x1, y1, x2, y2, label_es, conf, 
                                         color=(100, 100, 100), thickness=1)
                
                # Calcular total de obstáculos
                total_obstacles = debris_pothole_count + person_animal_count
                
                # Mostrar banner de alerta si se detectaron obstáculos
                if total_obstacles > 0:
                    # Crear texto detallado del tipo de obstáculos
                    if detection_types:
                        types_str = ', '.join([f"{count} {tipo}" for tipo, count in detection_types.items()])
                        alert_text = f"ALERTA: {types_str.upper()}"
                    else:
                        alert_text = f"ALERTA: {total_obstacles} OBSTACULO(S) DETECTADO(S)"
                    frame = draw_alert_banner(frame, alert_text)
                
                # Dibujar información de progreso en la parte inferior
                info_y = height - 85
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, info_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Obstaculos carretera: {debris_pothole_count} | Personas/Animales: {person_animal_count}", 
                          (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Total obstaculos: {total_obstacles} | Otros: {other_count}", 
                          (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Dibujar barra de progreso horizontal
                progress = int((frame_count / total_frames) * (width - 20))
                cv2.rectangle(frame, (10, height - 20), (width - 10, height - 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, height - 20), (10 + progress, height - 10), (0, 255, 0), -1)
                
                # Guardar frame en el video de salida si está habilitado
                if out:
                    out.write(frame)
                
                # Convertir el frame de BGR (OpenCV) a RGB (pygame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionar el frame si la ventana es más pequeña que el video
                if width != display_width:
                    frame_rgb = cv2.resize(frame_rgb, (display_width, display_height))
                
                # Ajustar la orientación del frame para pygame
                # OpenCV usa (filas, columnas), pygame usa (columnas, filas)
                frame_rgb = np.rot90(frame_rgb)
                frame_rgb = np.flipud(frame_rgb)
                
                # Convertir a superficie de pygame y mostrar
                surf = pygame.surfarray.make_surface(frame_rgb)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                # Controlar la velocidad de reproducción según los FPS configurados
                clock.tick(playback_fps)
    
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    
    finally:
        # Liberar todos los recursos
        cap.release()
        if out:
            out.release()
        pygame.quit()
        
        # Mostrar resumen final
        print()
        print("=" * 60)
        print("Procesamiento completado")
        print(f"Frames procesados: {frame_count}/{total_frames}")
        if args.save:
            print(f"Video guardado en: {output_path}")
        print("=" * 60)

if __name__ == "__main__":
    main()