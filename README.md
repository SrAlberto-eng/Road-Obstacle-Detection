# Road Obstacle Detection - Sistema de Detección de Obstáculos en Carretera

Sistema avanzado de detección de obstáculos en carreteras usando YOLOv8 con arquitectura dual-model para máxima cobertura y precisión.

---

## Estado del Proyecto

**MODELOS ENTRENADOS:**

### 1. Modelo Binario (Normal vs Obstacle)
- Dataset: 10,898 imágenes consolidadas
- Clases: normal (vehículos) vs obstacle (resto)
- Modelo: YOLOv8s con fine-tuning
- Resultado: 74.4% mAP50 general
- Limitación: Recall bajo (49.3%) en obstáculos debido a desbalance de datos

### 2. Modelo Debris/Pothole (Especializado)
- Dataset: 1,022 imágenes balanceadas (50-50)
- Clases: debris (escombros) y pothole (baches)
- Modelo: YOLOv8s con fine-tuning desde COCO
- Resultado: 79.2% mAP50, 82.8% Precision, 68.5% Recall
- Mejora significativa: +19.2% recall vs modelo binario

### 3. Sistema Dual-Model (Implementación Actual)
- Modelo 1: debris_pothole_v1 para obstáculos de carretera
- Modelo 2: YOLOv8s pre-entrenado COCO para personas/animales
- Cobertura completa: Detecta debris, potholes, personas, animales
- Sistema de alertas en tiempo real con visualización interactiva

---

## Datasets Disponibles

### Dataset Principal: road_obstacles_final (6 clases)
- Total: 10,898 imágenes
- Clases: vehicle, person, bicycle, animal, pothole, debris
- Splits: train/val/test
- Origen: Fusión de 4 datasets (animals, potholes, debris, self-driving)

### Dataset Balanceado: debris_pothole_balanced (2 clases)
- Total: 1,022 imágenes (511 debris + 511 pothole)
- Balanceo perfecto: 50-50%
- Splits: 80% train (817), 15% val (153), 5% test (52)
- Propósito: Entrenamiento especializado en defectos de carretera

---

## Inicio Rápido

### Configuración Inicial

**1. Activar Entorno Virtual**
```powershell
.\venv\Scripts\Activate.ps1
```

**2. Verificar Instalación**
```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

### Detección en Video (Método Recomendado)

**Ejecutar Sistema Dual-Model**

Opción A: Usando script batch (más fácil)
```cmd
run_detection.bat
```

Opción B: Línea de comandos
```powershell
.\venv\Scripts\Activate.ps1
python scripts/inference/predict_video_realtime.py `
  --video data/test.mp4 `
  --model results/training/debris_pothole_v1/weights/best.pt `
  --conf 0.65 `
  --conf-coco 0.65
```

**Controles:**
- ESPACIO: Pausar/Reanudar
- Q: Salir

**Características:**
- Detección en tiempo real con doble modelo
- Alertas visuales rojas para obstáculos
- Contadores de detección por tipo
- Barra de progreso integrada
- Información detallada en pantalla

### Procesamiento de Video sin GUI

Para procesar videos sin interfaz gráfica (guardar resultados):

```powershell
python scripts/inference/predict_video.py `
  --video data/test.mp4 `
  --model results/training/debris_pothole_v1/weights/best.pt `
  --conf 0.65 `
  --save
```

Salida: Video procesado en `runs/detect/predict/`

---

## Entrenamiento de Modelos

### Entrenar Modelo Debris/Pothole (Recomendado)

Este modelo especializado ofrece mejor rendimiento para obstáculos de carretera.

**1. Verificar Dataset Balanceado**
```powershell
python scripts/analysis/validate_yolo_dataset.py --data data/debris_pothole_balanced/data.yaml
```

**2. Entrenar**
```powershell
python scripts/training/train_binary_finetune.py `
  --datos data/debris_pothole_balanced/data.yaml `
  --epocas 100 `
  --batch 16 `
  --nombre debris_pothole_model `
  --congelar 10
```

**Configuración Óptima:**
- Modelo base: yolov8s.pt (pre-entrenado COCO)
- Freeze: 10 capas (mantiene conocimiento base)
- Learning rate: 0.01 (inicial)
- Batch: 16 (8 si hay problemas de memoria)
- Épocas: 100 (con early stopping patience=50)
- Tiempo estimado: 45-60 minutos (RTX 4060)

**3. Evaluar Resultados**
```powershell
python scripts/evaluation/evaluate_binary.py `
  --model results/training/debris_pothole_model/weights/best.pt `
  --data data/debris_pothole_balanced/data.yaml
```

**Métricas Esperadas:**
- mAP50: 75-80%
- Recall: 65-70%
- Precision: 80-85%

### Crear Nuevo Dataset Balanceado

Si necesitas modificar el balance de clases:

```powershell
python scripts/data/crear_dataset_debris_baches.py
```

Este script:
- Extrae debris y potholes del dataset consolidado
- Balancea a partes iguales (50-50%)
- Crea splits train/val/test (80/15/5)
- Genera data.yaml automáticamente

### Re-entrenar Modelo Binario

Para entrenar el modelo binario original (normal vs obstacle):

```powershell
python scripts/training/train_binary_finetune.py `
  --datos data/road_obstacles_binary/data.yaml `
  --epocas 100 `
  --nombre road_obstacles_binary_v3
```

Nota: Este modelo tiene limitaciones debido al desbalance de datos (5:1 ratio).

---

## Estructura del Proyecto

```
road_obstacle_detection/
├── data/                                   # Datasets
│   ├── road_obstacles_final/               # Dataset consolidado (6 clases)
│   │   ├── images/                         # Imágenes organizadas
│   │   │   ├── train/                      # 8,718 imágenes
│   │   │   ├── val/                        # 1,635 imágenes
│   │   │   └── test/                       # 545 imágenes
│   │   ├── labels/                         # Anotaciones YOLO
│   │   └── data.yaml                       # Configuración del dataset
│   ├── debris_pothole_balanced/            # Dataset balanceado (2 clases)
│   │   ├── images/
│   │   │   ├── train/                      # 817 imágenes
│   │   │   ├── val/                        # 153 imágenes
│   │   │   └── test/                       # 52 imágenes
│   │   ├── labels/
│   │   └── data.yaml
│   ├── road_obstacles_binary/              # Dataset binario (normal vs obstacle)
│   └── test.mp4                            # Video de prueba
├── docs/                                   # Documentación técnica
│   ├── EXPLICACION_TECNICA.md              # Explicación detallada del proyecto
│   ├── BDD100K_ANALYSIS.md                 # Análisis de datasets
│   ├── CONVERSION_SUMMARY.md               # Resumen de conversiones
│   ├── DATASET_BALANCE_REPORT.md           # Reportes de balance
│   └── DATASETS_RECOMMENDATIONS.md         # Recomendaciones
├── scripts/                                # Scripts del proyecto
│   ├── training/
│   │   └── train_binary_finetune.py        # Script principal de entrenamiento
│   ├── inference/
│   │   ├── predict_video.py                # Procesamiento de video
│   │   └── predict_video_realtime.py       # Demo interactiva (dual-model)
│   ├── data/
│   │   └── crear_dataset_debris_baches.py  # Creación de dataset balanceado
│   ├── evaluation/
│   │   └── evaluate_binary.py              # Evaluación de modelos
│   ├── analysis/
│   │   ├── validate_yolo_dataset.py        # Validación de datasets
│   │   └── analyze_dataset_balance.py      # Análisis de balance
│   ├── preprocessing/
│   │   ├── coco_to_yolo.py                 # Conversión de formatos
│   │   └── bdd100k_to_yolo_direct.py
│   └── utils/                              # Utilidades
├── results/                                # Resultados de entrenamiento
│   └── training/
│       ├── road_obstacles_binary_v1/       # Modelo binario
│       │   └── weights/best.pt
│       ├── debris_pothole_v1/              # Modelo especializado
│       │   ├── weights/best.pt             # Mejor modelo
│       │   ├── results.csv                 # Métricas por época
│       │   └── confusion_matrix.png        # Matriz de confusión
│       └── road_obstacles_v1/              # Modelo original (6 clases)
├── requirements.txt                        # Dependencias Python
├── run_detection.bat                       # Script rápido de ejecución
└── README.md                               # Este archivo
```

---

## Documentación Técnica

### Documentos Principales
- **[docs/EXPLICACION_TECNICA.md](docs/EXPLICACION_TECNICA.md)** - Explicación completa del proyecto
  - Arquitectura de YOLOv8 (Backbone, Neck, Head)
  - Fine-tuning y transfer learning
  - Dataset COCO y formatos de anotación
  - Descripción detallada de scripts
  - Dificultades encontradas y soluciones
  - Hiperparámetros explicados
  - Métricas de evaluación (mAP, Precision, Recall)

### Documentos Complementarios
- **[docs/DATASET_BALANCE_REPORT.md](docs/DATASET_BALANCE_REPORT.md)** - Análisis de balance de clases
- **[docs/CONVERSION_SUMMARY.md](docs/CONVERSION_SUMMARY.md)** - Conversiones COCO a YOLO
- **[docs/BDD100K_ANALYSIS.md](docs/BDD100K_ANALYSIS.md)** - Análisis de datasets disponibles
- **[docs/DATASETS_RECOMMENDATIONS.md](docs/DATASETS_RECOMMENDATIONS.md)** - Recomendaciones de datasets

---

## Scripts Principales

### Entrenamiento
**`scripts/training/train_binary_finetune.py`**
- Script principal de entrenamiento con fine-tuning
- Soporta congelamiento de capas (freeze)
- Data augmentation configurado para dashcam
- Early stopping automático
- Métricas detalladas por época
- Todo el código en español

**Argumentos:**
- `--datos`: Ruta al data.yaml
- `--modelo`: Modelo base (yolov8s.pt por defecto)
- `--epocas`: Número de épocas (100 por defecto)
- `--batch`: Tamaño del batch (8 por defecto)
- `--congelar`: Capas a congelar (10 por defecto)
- `--dispositivo`: cuda/cpu (automático por defecto)

### Inferencia
**`scripts/inference/predict_video_realtime.py`**
- Sistema dual-model con visualización interactiva
- Detecta debris, potholes, personas y animales
- Alertas visuales en tiempo real
- Controles interactivos (pausa, salir)
- Información detallada en pantalla
- Usa pygame para visualización

**Argumentos:**
- `--video`: Ruta al video
- `--model`: Modelo debris/pothole
- `--model-coco`: Modelo COCO (yolov8s.pt por defecto)
- `--conf`: Umbral para debris/pothole (0.75 por defecto)
- `--conf-coco`: Umbral para COCO (0.5 por defecto)
- `--save`: Guardar video procesado
- `--fps`: FPS de reproducción

**`scripts/inference/predict_video.py`**
- Procesamiento de video sin GUI
- Guarda resultados automáticamente
- Más rápido que modo interactivo
- Ideal para procesamiento batch

### Datos
**`scripts/data/crear_dataset_debris_baches.py`**
- Crea dataset balanceado de debris y potholes
- Extrae clases específicas del dataset consolidado
- Balancea automáticamente (50-50%)
- Genera splits train/val/test
- Crea data.yaml y README.md

### Análisis
**`scripts/analysis/validate_yolo_dataset.py`**
- Valida estructura de datasets YOLO
- Verifica integridad de imágenes y etiquetas
- Detecta archivos corruptos
- Estadísticas por clase

**`scripts/analysis/analyze_dataset_balance.py`**
- Analiza balance de clases
- Genera reportes detallados
- Calcula ratios y distribuciones
- Visualizaciones con matplotlib

### Evaluación
**`scripts/evaluation/evaluate_binary.py`**
- Evalúa modelos entrenados
- Calcula métricas completas (mAP, Precision, Recall)
- Genera matriz de confusión
- Resultados por clase y globales

---

## Workflow del Proyecto

### Pipeline de Entrenamiento

1. **Preparación de Datos**
   ```
   Dataset Consolidado → Crear Dataset Balanceado → Validar Estructura
   road_obstacles_final    crear_dataset_*.py       validate_yolo_dataset.py
   ```

2. **Entrenamiento**
   ```
   Modelo Pre-entrenado → Fine-tuning → Validación → Modelo Final
   yolov8s.pt (COCO)      train_*.py    evaluate_*   best.pt
   ```

3. **Evaluación y Uso**
   ```
   Modelo Entrenado → Evaluación → Inferencia → Resultados
   best.pt            metrics        predict_*    video procesado
   ```

### Ciclo de Mejora

```
Entrenar → Evaluar → Analizar Errores → Ajustar Datos/Hiperparámetros → Repetir
```

**Métricas Clave a Monitorear:**
- mAP50: Calidad general del modelo
- Recall: Capacidad de detectar todos los obstáculos (crítico)
- Precision: Reducir falsas alarmas
- Balance de datos: Ratio entre clases

---

## Métricas y Rendimiento

### Resultados Obtenidos

**Modelo Debris/Pothole (debris_pothole_v1)**
```
Métrica General:
- mAP50: 79.2%
- Precision: 82.8%
- Recall: 68.5%

Por Clase:
Debris:
  - Precision: 85.9%
  - Recall: 81.8%
  - mAP50: 88.8%

Pothole:
  - Precision: 79.6%
  - Recall: 55.2%
  - mAP50: 69.6%
```

**Modelo Binario (road_obstacles_binary_v1)**
```
Clase Obstacle:
  - mAP50: 55.3%
  - Precision: 72.1%
  - Recall: 49.3% (bajo debido a desbalance)

Clase Normal:
  - mAP50: 93.5%
  - Precision: 89.7%
  - Recall: 82.6%
```

### Rendimiento de Inferencia

**Hardware: RTX 4060 Laptop 8GB**
- Velocidad de procesamiento: ~60 FPS
- Latencia por frame: ~16ms
- VRAM utilizada: 3.6-4.0 GB (batch 16)
- Procesamiento dual-model: ~30 FPS

**Variantes de Modelo:**
- YOLOv8n: 120+ FPS, menor precisión
- YOLOv8s: 60-80 FPS, balance óptimo (usado)
- YOLOv8m: 40-50 FPS, mayor precisión

### Mejoras Logradas

**Modelo Balanceado vs Binario:**
- Recall: +19.2% (49.3% → 68.5%)
- mAP50: +5.2% (74.0% → 79.2%)
- Precision: +6.8% (76.0% → 82.8%)

**Ventajas del Approach Dual-Model:**
- Cobertura completa: debris, potholes, personas, animales
- Mantiene conocimiento COCO (80 clases)
- Especialización en obstáculos de carretera
- Flexibilidad en umbrales de confianza por tipo

---

## Solución de Problemas

### Problemas Comunes

**1. GPU no detectada**
```powershell
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si es False, reinstalar PyTorch con CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Error de CUDA Out of Memory**
```powershell
# Reducir batch size en entrenamiento
python scripts/training/train_binary_finetune.py --batch 8  # o --batch 4
```

**3. Multiprocessing Error en Windows**
- El script ya está configurado con `workers=0` para evitar este problema
- Si persiste, verificar que el script tenga la protección `if __name__ == "__main__":`

**4. OpenCV no muestra ventanas**
- El proyecto usa pygame en lugar de OpenCV GUI
- Si necesitas OpenCV con GUI:
```powershell
pip uninstall opencv-python-headless -y
pip install opencv-python
```

**5. Pygame no instalado**
```powershell
pip install pygame
```

**6. Video no se encuentra**
- Verificar ruta del video con path absoluto
- Colocar video en carpeta `data/` para rutas relativas
- Formato soportado: .mp4, .avi, .mov

**7. Modelo no se encuentra**
- Verificar que el entrenamiento haya completado
- Path correcto: `results/training/[nombre_modelo]/weights/best.pt`
- Usar path absoluto si hay problemas con rutas relativas

**8. Recall muy bajo (< 50%)**
- Problema: Dataset desbalanceado
- Solución: Usar dataset balanceado `debris_pothole_balanced`
- Alternativa: Ajustar class weights en entrenamiento

**9. Muchas falsas alarmas (Precision baja)**
- Aumentar umbral de confianza: `--conf 0.75` o `--conf 0.8`
- Re-entrenar con más data augmentation
- Verificar calidad de anotaciones en dataset

**10. Entrenamiento muy lento**
- Verificar uso de GPU: debe mostrar "CUDA:0" al iniciar
- Reducir resolución de imagen: `--tamaño-img 416` (default 640)
- Usar modelo más pequeño: yolov8n.pt en lugar de yolov8s.pt

---

## Requisitos del Sistema

### Hardware Mínimo
- CPU: Intel i5 o equivalente
- RAM: 8 GB
- Almacenamiento: 10 GB libres

### Hardware Recomendado
- CPU: Intel i7 o equivalente
- RAM: 16 GB
- GPU: NVIDIA con 6+ GB VRAM (RTX 3060 o superior)
- Almacenamiento: 20 GB libres SSD

### Software
- Sistema Operativo: Windows 10/11, Linux, macOS
- Python: 3.8 - 3.13
- CUDA: 11.8 (para uso de GPU NVIDIA)
- Git (para clonar repositorio)

### Dependencias Python
Ver `requirements.txt` para lista completa:
- torch >= 2.0.0 (con CUDA para GPU)
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- PyYAML >= 6.0
- matplotlib >= 3.7.0
- pygame >= 2.6.0 (para visualización)

---

## Instalación

### 1. Clonar Repositorio
```bash
git clone https://github.com/usuario/road_obstacle_detection.git
cd road_obstacle_detection
```

### 2. Crear Entorno Virtual
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias
```powershell
# PyTorch con CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Resto de dependencias
pip install -r requirements.txt
```

### 4. Verificar Instalación
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Uso Básico

### Ejecutar Detección en Video
```powershell
# Método más sencillo
run_detection.bat

# O manualmente
.\venv\Scripts\Activate.ps1
python scripts/inference/predict_video_realtime.py --video data/test.mp4 --model results/training/debris_pothole_v1/weights/best.pt
```

### Entrenar Modelo Propio
```powershell
python scripts/training/train_binary_finetune.py --datos data/debris_pothole_balanced/data.yaml --epocas 100
```

### Evaluar Modelo
```powershell
python scripts/evaluation/evaluate_binary.py --model results/training/[modelo]/weights/best.pt
```

---

## Próximas Mejoras

**Planeadas:**
- Exportación a ONNX y TensorRT para dispositivos edge
- Soporte para detección de señales de tráfico
- Modo batch para procesamiento de múltiples videos
- API REST para integración con otros sistemas
- Dashboard web para visualización de métricas

**Áreas de Investigación:**
- Aumentar dataset con más imágenes de baches y escombros
- Probar arquitecturas más recientes (YOLOv9, YOLOv10)
- Implementar tracking para seguimiento de obstáculos
- Validación en condiciones adversas (lluvia, noche, niebla)

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crear branch para nueva funcionalidad
3. Hacer commit de cambios
4. Enviar pull request con descripción detallada

---

## Licencia

Este proyecto es para fines educativos y de investigación.

---

## Contacto y Soporte

Para preguntas o problemas:
1. Revisar sección de Solución de Problemas
2. Consultar documentación en `docs/`
3. Abrir issue en GitHub con detalles del problema

---

## Créditos

**Datasets Utilizados:**
- COCO Dataset (Common Objects in Context)
- Datasets públicos de Roboflow (animals, potholes, debris)
- BDD100K (análisis de compatibilidad)

**Frameworks y Librerías:**
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Pygame

---

**Versión del Proyecto:** 2.0  
**Última Actualización:** Noviembre 2025  
**Estado:** En desarrollo activo
