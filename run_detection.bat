@echo off
REM Script para ejecutar detección de obstáculos en tiempo real
REM Usa modelo de debris/potholes + COCO para personas/animales

echo ========================================
echo Detector de Obstaculos - Video en Tiempo Real
echo ========================================
echo.

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Ejecutar script de detección
python scripts/inference/predict_video_realtime.py ^
  --video data/test.mp4 ^
  --model models/trained/debris_pothole_v1/weights/best.pt ^
  --conf 0.65 ^
  --conf-coco 0.65

echo.
echo ========================================
echo Proceso completado
echo ========================================
pause
