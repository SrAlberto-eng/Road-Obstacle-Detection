"""
Script para generar diagrama del pipeline de detección de obstáculos.
Crea diagramas visuales para incluir en el informe académico.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

def create_pipeline_diagram(output_path):
    """
    Crea diagrama del pipeline completo de detección.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configurar figura
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colores
    color_input = '#E3F2FD'      # Azul claro - Entrada
    color_preprocess = '#FFF9C4'  # Amarillo claro - Preprocesamiento
    color_model = '#C8E6C9'       # Verde claro - Modelo
    color_postprocess = '#FFCCBC' # Naranja claro - Post-procesamiento
    color_output = '#F8BBD0'      # Rosa claro - Salida
    
    # Título
    ax.text(5, 11.5, 'Pipeline de Detección de Obstáculos en Carreteras', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(5, 11, 'Road Obstacle Detection System - YOLOv8', 
            ha='center', va='top', fontsize=12, style='italic')
    
    y_pos = 10
    
    # 1. ENTRADA - Imagen de prueba
    box1 = FancyBboxPatch((1, y_pos-0.8), 8, 0.8, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_input, linewidth=2)
    ax.add_patch(box1)
    ax.text(5, y_pos-0.4, '1. ENTRADA', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.65, 'Imagen del conjunto de prueba (test)', 
            ha='center', va='center', fontsize=9)
    
    # Flecha
    arrow1 = FancyArrowPatch((5, y_pos-0.8), (5, y_pos-1.5),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    y_pos -= 1.5
    
    # 2. PREPROCESAMIENTO
    box2 = FancyBboxPatch((1, y_pos-1.2), 8, 1.2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_preprocess, linewidth=2)
    ax.add_patch(box2)
    ax.text(5, y_pos-0.3, '2. PREPARACIÓN DE LA IMAGEN', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.6, '• Cambiar tamaño a 640×640 píxeles', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos-0.85, '• Ajustar brillo y contraste', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos-1.05, '• Preparar formato para el modelo', 
            ha='center', va='center', fontsize=9)
    
    # Flecha
    arrow2 = FancyArrowPatch((5, y_pos-1.2), (5, y_pos-1.9),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    y_pos -= 1.9
    
    # 3. MODELO YOLO
    box3 = FancyBboxPatch((1, y_pos-2.5), 8, 2.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_model, linewidth=2)
    ax.add_patch(box3)
    ax.text(5, y_pos-0.4, '3. MODELO YOLOv8 (Inteligencia Artificial)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Sub-componentes del modelo
    ax.text(5, y_pos-0.8, 'Paso 1: Analizar la imagen completa', 
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, y_pos-1.1, 'Paso 2: Buscar patrones de objetos', 
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(5, y_pos-1.4, 'Paso 3: Identificar ubicación y tipo de obstáculos', 
            ha='center', va='center', fontsize=9, style='italic')
    
    ax.text(5, y_pos-1.8, 'Resultado: Lista de objetos detectados', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, y_pos-2.1, '(Cada objeto con ubicación, tipo y confianza)', 
            ha='center', va='center', fontsize=8)
    
    # Flecha
    arrow3 = FancyArrowPatch((5, y_pos-2.5), (5, y_pos-3.2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    y_pos -= 3.2
    
    # 4. POST-PROCESAMIENTO
    box4 = FancyBboxPatch((1, y_pos-1.5), 8, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_postprocess, linewidth=2)
    ax.add_patch(box4)
    ax.text(5, y_pos-0.3, '4. FILTRADO DE RESULTADOS', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.6, '• Eliminar detecciones con baja confianza (< 25%)', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos-0.9, '• Quitar detecciones duplicadas', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos-1.2, '• Ajustar coordenadas al tamaño original', 
            ha='center', va='center', fontsize=9)
    
    # Flecha
    arrow4 = FancyArrowPatch((5, y_pos-1.5), (5, y_pos-2.2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    y_pos -= 2.2
    
    # 5. SALIDA
    box5 = FancyBboxPatch((1, y_pos-1.2), 8, 1.2, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_output, linewidth=2)
    ax.add_patch(box5)
    ax.text(5, y_pos-0.3, '5. RESULTADO FINAL', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(5, y_pos-0.6, 'Imagen con rectángulos sobre obstáculos:', 
            ha='center', va='center', fontsize=9)
    ax.text(5, y_pos-0.9, '[Tipo de objeto, nivel de certeza, ubicación]', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # Información adicional en el lado
    ax.text(9.5, 8, 'Tipos de Obstáculos:', ha='left', va='top', 
            fontsize=9, fontweight='bold')
    classes = ['• Vehículos', '• Personas', '• Bicicletas', '• Animales', '• Baches', '• Escombros']
    for i, cls in enumerate(classes):
        ax.text(9.5, 7.6-i*0.3, cls, ha='left', va='top', fontsize=8)
    
    ax.text(9.5, 5, 'Modelo YOLOv8:', ha='left', va='top', 
            fontsize=9, fontweight='bold')
    ax.text(9.5, 4.7, 'Archivo: yolov8s.pt', ha='left', va='top', fontsize=8)
    ax.text(9.5, 4.4, 'Precisión: 59.85%', ha='left', va='top', fontsize=8)
    ax.text(9.5, 4.1, 'Velocidad: Rápida', ha='left', va='top', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar
    output_file = output_path / 'pipeline_deteccion.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagrama guardado: {output_file}")
    
    plt.close()

def create_inference_flowchart(output_path):
    """
    Crea diagrama de flujo del proceso de inferencia.
    """
    output_path = Path(output_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Título
    ax.text(5, 15.5, 'Proceso de Detección de Obstáculos', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    y_pos = 15
    
    # Función para crear caja de proceso
    def add_process_box(y, text, color='#E3F2FD'):
        box = FancyBboxPatch((2, y-0.6), 6, 0.6, 
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, y-0.3, text, ha='center', va='center', fontsize=9)
        return y - 0.6
    
    # Función para crear caja de decisión
    def add_decision_box(y, text):
        # Rombo para decisión
        points = [(5, y), (7, y-0.4), (5, y-0.8), (3, y-0.4)]
        polygon = mpatches.Polygon(points, closed=True, 
                                  edgecolor='black', facecolor='#FFF9C4', linewidth=1.5)
        ax.add_patch(polygon)
        ax.text(5, y-0.4, text, ha='center', va='center', fontsize=8)
        return y - 0.8
    
    # Función para flecha
    def add_arrow(y_from, y_to):
        arrow = FancyArrowPatch((5, y_from), (5, y_to),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    # Flujo
    y_pos = add_process_box(y_pos, 'INICIO: Seleccionar imagen del conjunto test', '#C8E6C9')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Cambiar tamaño a 640×640 píxeles')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Ajustar brillo y contraste')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Preparar formato para el modelo')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Cargar modelo YOLOv8 (GPU)', '#FFCCBC')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Analizar imagen con IA', '#C8E6C9')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Obtener posibles obstáculos')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_decision_box(y_pos, '¿Confianza > 25%?')
    
    # Rama NO
    ax.text(7.5, y_pos+0.4, 'NO', ha='left', va='center', fontsize=8, style='italic')
    arrow_no = FancyArrowPatch((7, y_pos+0.4), (8.5, y_pos+0.4),
                              arrowstyle='->', mutation_scale=15, 
                              linewidth=1.5, color='red')
    ax.add_patch(arrow_no)
    ax.text(8.7, y_pos+0.4, 'Descartar', ha='left', va='center', fontsize=7, color='red')
    
    # Rama SÍ
    ax.text(5, y_pos-0.15, 'SÍ', ha='center', va='top', fontsize=8, style='italic')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Eliminar detecciones duplicadas')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Ajustar coordenadas al tamaño original')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_decision_box(y_pos, '¿Hay obstáculos?')
    
    # Rama NO
    ax.text(7.5, y_pos+0.4, 'NO', ha='left', va='center', fontsize=8, style='italic')
    arrow_no2 = FancyArrowPatch((7, y_pos+0.4), (8.5, y_pos+0.4),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='orange')
    ax.add_patch(arrow_no2)
    ax.text(8.7, y_pos+0.4, 'Sin obstáculos', ha='left', va='center', fontsize=7, color='orange')
    
    # Rama SÍ
    ax.text(5, y_pos-0.15, 'SÍ', ha='center', va='top', fontsize=8, style='italic')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Dibujar rectángulos sobre obstáculos')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'Guardar lista de obstáculos detectados', '#F8BBD0')
    add_arrow(y_pos, y_pos-0.4)
    y_pos -= 0.4
    
    y_pos = add_process_box(y_pos, 'FIN: Imagen con obstáculos marcados', '#C8E6C9')
    
    # Leyenda
    ax.text(0.5, 2, 'Leyenda:', ha='left', va='top', fontsize=9, fontweight='bold')
    legend_items = [
        ('Proceso normal', '#E3F2FD'),
        ('Uso de GPU', '#FFCCBC'),
        ('Decisión', '#FFF9C4'),
        ('Inicio/Fin', '#C8E6C9'),
        ('Resultado', '#F8BBD0')
    ]
    for i, (label, color) in enumerate(legend_items):
        small_box = FancyBboxPatch((0.5, 1.5-i*0.3), 0.3, 0.2, 
                                  boxstyle="round,pad=0.02", 
                                  edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(small_box)
        ax.text(0.9, 1.6-i*0.3, label, ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar
    output_file = output_path / 'flujo_inferencia.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagrama guardado: {output_file}")
    
    plt.close()

def create_architecture_diagram(output_path):
    """
    Crea diagrama de arquitectura del sistema completo.
    """
    output_path = Path(output_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Título
    ax.text(7, 11.5, 'Arquitectura del Sistema de Detección de Obstáculos', 
            ha='center', va='top', fontsize=15, fontweight='bold')
    
    # Capa 1: Hardware
    box_hw = FancyBboxPatch((0.5, 9), 13, 1.5, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#E0E0E0', linewidth=2)
    ax.add_patch(box_hw)
    ax.text(7, 10.2, 'HARDWARE (Computadora)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(3, 9.7, 'Tarjeta Gráfica\nRTX 4060 (8GB)', ha='center', va='center', fontsize=9)
    ax.text(7, 9.7, 'Procesador\n16GB Memoria', ha='center', va='center', fontsize=9)
    ax.text(11, 9.7, 'Disco Duro\nModelo: 6MB', ha='center', va='center', fontsize=9)
    
    # Capa 2: Framework
    box_fw = FancyBboxPatch((0.5, 6.8), 13, 1.5, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(box_fw)
    ax.text(7, 8, 'SOFTWARE BASE (Librerías)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(3, 7.5, 'PyTorch\n(Inteligencia Artificial)', ha='center', va='center', fontsize=9)
    ax.text(7, 7.5, 'Ultralytics\n(Framework YOLO)', ha='center', va='center', fontsize=9)
    ax.text(11, 7.5, 'OpenCV\n(Procesamiento Imágenes)', ha='center', va='center', fontsize=9)
    
    # Capa 3: Modelo
    box_model = FancyBboxPatch((0.5, 4), 13, 2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(box_model)
    ax.text(7, 5.6, 'MODELO (YOLOv8)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(3, 5.1, 'Análisis de Imagen\n(Paso 1)', ha='center', va='center', fontsize=9)
    ax.text(7, 5.1, 'Búsqueda de Patrones\n(Paso 2)', ha='center', va='center', fontsize=9)
    ax.text(11, 5.1, 'Identificación\n(Paso 3)', ha='center', va='center', fontsize=9)
    ax.text(7, 4.5, 'Archivo: yolov8s.pt (22 MB) | 6 tipos de obstáculos | Precisión: 59.85%', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # Capa 4: Aplicación
    box_app = FancyBboxPatch((0.5, 1.5), 13, 1.8, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#FFCCBC', linewidth=2)
    ax.add_patch(box_app)
    ax.text(7, 3, 'PROCESAMIENTO DE IMÁGENES', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(2.5, 2.4, 'Entrada\nImagen Test', ha='center', va='center', fontsize=9)
    ax.text(5.5, 2.4, 'Preparación\nAjustar Tamaño', ha='center', va='center', fontsize=9)
    ax.text(8.5, 2.4, 'Detección\nAnálisis IA', ha='center', va='center', fontsize=9)
    ax.text(11.5, 2.4, 'Resultado\nImagen Marcada', ha='center', va='center', fontsize=9)
    ax.text(7, 1.8, 'Programas: entrenar, predecir, evaluar', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # Capa 5: Interfaz
    box_ui = FancyBboxPatch((0.5, 0), 13, 1, 
                           boxstyle="round,pad=0.1", 
                           edgecolor='black', facecolor='#F8BBD0', linewidth=2)
    ax.add_patch(box_ui)
    ax.text(7, 0.7, 'INTERACCIÓN CON EL USUARIO', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(4, 0.3, 'Línea de Comandos', ha='center', va='center', fontsize=9)
    ax.text(7, 0.3, 'Archivos Resultados', ha='center', va='center', fontsize=9)
    ax.text(10, 0.3, 'Imágenes Marcadas', ha='center', va='center', fontsize=9)
    
    # Flechas entre capas
    for y in [9, 6.8, 4, 1.5]:
        arrow = FancyArrowPatch((7, y), (7, y-0.6),
                               arrowstyle='<->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    
    # Guardar
    output_file = output_path / 'arquitectura_sistema.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Diagrama guardado: {output_file}")
    
    plt.close()

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("GENERACIÓN DE DIAGRAMAS DEL PIPELINE DE DETECCIÓN")
    print("="*70)
    
    output_path = Path('docs/evidencias/diagramas')
    
    print("\n1. Generando diagrama del pipeline de detección...")
    create_pipeline_diagram(output_path)
    
    print("\n2. Generando diagrama de flujo de inferencia...")
    create_inference_flowchart(output_path)
    
    print("\n3. Generando diagrama de arquitectura del sistema...")
    create_architecture_diagram(output_path)
    
    print("\n" + "="*70)
    print("DIAGRAMAS GENERADOS EXITOSAMENTE")
    print("="*70)
    print(f"\nArchivos guardados en: {output_path}")
    print("  1. pipeline_deteccion.png - Pipeline completo")
    print("  2. flujo_inferencia.png - Flujo de inferencia detallado")
    print("  3. arquitectura_sistema.png - Arquitectura por capas")
    print("\nEstos diagramas pueden ser incluidos en tu informe académico.")
    print("="*70)

if __name__ == '__main__':
    main()
