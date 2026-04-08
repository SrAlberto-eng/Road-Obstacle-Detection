"""
Script de inferencia en tiempo real con webcam
Road Obstacle Detection - Live Demo
"""

import argparse
import cv2
import torch
from ultralytics import YOLO
import time

def main():
    parser = argparse.ArgumentParser(description='Road Obstacle Detection - Webcam en tiempo real')
    parser.add_argument('--weights', type=str, required=True,
                        help='Ruta al modelo entrenado (best.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Umbral de confianza (default: 0.5)')
    parser.add_argument('--camera', type=int, default=0,
                        help='ID de la cámara (default: 0)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Tamaño de imagen (default: 640)')
    
    args = parser.parse_args()
    
    # Verificar GPU
    print("=" * 60)
    print("ROAD OBSTACLE DETECTION - TIEMPO REAL")
    print("=" * 60)
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Cargar modelo
    print(f"\nCargando modelo: {args.weights}")
    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Clases del modelo
    classes = model.names
    print(f"\nClases detectables: {list(classes.values())}")
    
    # Colores para cada clase (BGR)
    colors = {
        0: (0, 255, 0),      # vehicle - verde
        1: (255, 0, 0),      # person - azul
        2: (0, 255, 255),    # two_wheeler - amarillo
        3: (255, 0, 255),    # animal - magenta
        4: (0, 165, 255),    # pothole - naranja
        5: (0, 0, 255),      # debris - rojo
    }
    
    # Inicializar cámara
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {args.camera}")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "=" * 60)
    print("CÁMARA INICIADA")
    print("Presiona 'q' para salir")
    print("=" * 60 + "\n")
    
    # Variables para FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo leer el frame")
            break
        
        # Realizar predicción
        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
            stream=False
        )
        
        # Procesar resultados
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Obtener coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = classes[cls]
                
                # Color según clase
                color = colors.get(cls, (255, 255, 255))
                
                # Dibujar bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiqueta
                label = f"{class_name}: {conf:.2f}"
                
                # Fondo para el texto
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, 
                    (x1, y1 - text_height - baseline - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - baseline - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), 
                    2
                )
        
        # Calcular FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Mostrar FPS
        cv2.putText(
            frame, 
            f"FPS: {fps}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Mostrar frame
        cv2.imshow('Road Obstacle Detection', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("CÁMARA CERRADA")
    print("=" * 60)

if __name__ == '__main__':
    main()
