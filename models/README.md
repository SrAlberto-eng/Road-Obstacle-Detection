# Models Directory Structure

## Organización

```
models/
├── pretrained/          # Modelos pre-entrenados de Ultralytics
│   ├── yolov8n.pt      # YOLOv8 nano (3.2M params, 6.25 MB)
│   ├── yolov8s.pt      # YOLOv8 small (11.2M params, 21.54 MB) ← USADO EN TRAIN.PY
│   └── yolo11n.pt      # YOLO11 nano (5.35 MB)
│
├── checkpoints/         # Checkpoints durante entrenamiento (se creará automáticamente)
│
└── trained/            # Modelos finales entrenados (copiar best.pt aquí después del entrenamiento)
```

## Uso

### Durante Entrenamiento

El script `train.py` usa: `models/pretrained/yolov8s.pt`

### Después del Entrenamiento

Los resultados se guardan en:
```
results/training/road_obstacles_v1/weights/
├── best.pt    ← Mejor modelo (copiar a models/trained/)
└── last.pt    ← Último checkpoint
```

### Para Inferencia

Usa el mejor modelo entrenado:
```python
from ultralytics import YOLO
model = YOLO('models/trained/road_obstacles_best.pt')
```

## Modelos Disponibles

| Modelo | Parámetros | Tamaño | Velocidad | Precisión | Uso |
|--------|-----------|--------|-----------|-----------|-----|
| yolov8n.pt | 3.2M | 6.25 MB | Rápido | Baja | Tests rápidos |
| yolov8s.pt | 11.2M | 21.54 MB | Medio | Media-Alta | **Entrenamiento principal** |
| yolo11n.pt | ~3M | 5.35 MB | Rápido | Media | Experimenta YOLO11 |

## Notas

- Los modelos pre-entrenados se descargan automáticamente la primera vez
- Los checkpoints se guardan cada 10 épocas durante el entrenamiento
- El mejor modelo (`best.pt`) se selecciona automáticamente basado en mAP50
