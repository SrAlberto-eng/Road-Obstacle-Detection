"""
Script para generar evidencia completa del entrenamiento realizado.
Extrae información de results.csv, args.yaml y genera resumen en formato académico.
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def read_training_args(results_path):
    """Lee los argumentos de entrenamiento desde args.yaml"""
    args_file = results_path / 'args.yaml'
    
    if not args_file.exists():
        return None
    
    with open(args_file, 'r') as f:
        args = yaml.safe_load(f)
    
    return args

def read_training_results(results_path):
    """Lee los resultados del entrenamiento desde results.csv"""
    results_file = results_path / 'results.csv'
    
    if not results_file.exists():
        return None
    
    # Leer CSV quitando espacios en los nombres de columnas
    df = pd.read_csv(results_file)
    df.columns = df.columns.str.strip()
    
    return df

def format_duration(seconds):
    """Formatea duración en formato legible"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def generate_evidence_report(results_path, output_path):
    """
    Genera reporte completo de evidencia del entrenamiento.
    
    Args:
        results_path: Path al directorio de resultados (ej: results/training/road_obstacles_v1)
        output_path: Path donde guardar el reporte
    """
    results_path = Path(results_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERACIÓN DE EVIDENCIA DE ENTRENAMIENTO")
    print("="*70)
    
    # 1. Leer argumentos de entrenamiento
    print("\n1. Leyendo configuración de entrenamiento...")
    args = read_training_args(results_path)
    
    if args is None:
        print(f"ERROR: No se encontró args.yaml en {results_path}")
        return
    
    # 2. Leer resultados
    print("2. Leyendo resultados de entrenamiento...")
    df = read_training_results(results_path)
    
    if df is None:
        print(f"ERROR: No se encontró results.csv en {results_path}")
        return
    
    # 3. Calcular métricas clave
    print("3. Calculando métricas clave...")
    
    total_epochs = len(df)
    
    # Encontrar mejor época
    best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
    best_epoch = df.loc[best_epoch_idx]
    
    # Métricas finales
    final_metrics = df.iloc[-1]
    
    # 4. Generar reporte en texto
    print("4. Generando reporte de evidencia...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EVIDENCIA DE ENTRENAMIENTO - ROAD OBSTACLE DETECTION")
    report_lines.append("="*80)
    report_lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Directorio de resultados: {results_path}")
    report_lines.append("")
    
    # Sección 1: Configuración del Entrenamiento
    report_lines.append("="*80)
    report_lines.append("1. CONFIGURACIÓN DEL ENTRENAMIENTO")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("1.1 Modelo y Dataset:")
    report_lines.append(f"  - Modelo base: {args.get('model', 'N/A')}")
    report_lines.append(f"  - Dataset: {args.get('data', 'N/A')}")
    report_lines.append(f"  - Clases: {args.get('nc', 'N/A')}")
    report_lines.append(f"  - Nombres: {args.get('names', 'N/A')}")
    report_lines.append("")
    
    report_lines.append("1.2 Hiperparámetros de Entrenamiento:")
    report_lines.append(f"  - Épocas: {args.get('epochs', 'N/A')}")
    report_lines.append(f"  - Batch size: {args.get('batch', 'N/A')}")
    report_lines.append(f"  - Tamaño de imagen: {args.get('imgsz', 'N/A')}")
    report_lines.append(f"  - Device: {args.get('device', 'N/A')}")
    report_lines.append(f"  - Optimizer: {args.get('optimizer', 'N/A')}")
    report_lines.append(f"  - Learning rate inicial: {args.get('lr0', 'N/A')}")
    report_lines.append(f"  - Learning rate final: {args.get('lrf', 'N/A')}")
    report_lines.append(f"  - Weight decay: {args.get('weight_decay', 'N/A')}")
    report_lines.append(f"  - Patience (early stopping): {args.get('patience', 'N/A')}")
    report_lines.append("")
    
    report_lines.append("1.3 Data Augmentation:")
    report_lines.append(f"  - HSV-H: {args.get('hsv_h', 'N/A')}")
    report_lines.append(f"  - HSV-S: {args.get('hsv_s', 'N/A')}")
    report_lines.append(f"  - HSV-V: {args.get('hsv_v', 'N/A')}")
    report_lines.append(f"  - Degrees: {args.get('degrees', 'N/A')}")
    report_lines.append(f"  - Translate: {args.get('translate', 'N/A')}")
    report_lines.append(f"  - Scale: {args.get('scale', 'N/A')}")
    report_lines.append(f"  - Shear: {args.get('shear', 'N/A')}")
    report_lines.append(f"  - FlipLR: {args.get('fliplr', 'N/A')}")
    report_lines.append(f"  - Mosaic: {args.get('mosaic', 'N/A')}")
    report_lines.append(f"  - Mixup: {args.get('mixup', 'N/A')}")
    report_lines.append(f"  - Copy-paste: {args.get('copy_paste', 'N/A')}")
    report_lines.append("")
    
    # Sección 2: Resultados del Entrenamiento
    report_lines.append("="*80)
    report_lines.append("2. RESULTADOS DEL ENTRENAMIENTO")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("2.1 Información General:")
    report_lines.append(f"  - Total de épocas ejecutadas: {total_epochs}")
    report_lines.append(f"  - Mejor época: {int(best_epoch['epoch']) + 1}")
    report_lines.append(f"  - Early stopping activado: {'Sí' if total_epochs < args.get('epochs', 0) else 'No'}")
    report_lines.append("")
    
    report_lines.append("2.2 Métricas del Mejor Modelo (Época {}):".format(int(best_epoch['epoch']) + 1))
    report_lines.append(f"  - mAP50: {best_epoch['metrics/mAP50(B)']:.4f}")
    report_lines.append(f"  - mAP50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")
    report_lines.append(f"  - Precision: {best_epoch['metrics/precision(B)']:.4f}")
    report_lines.append(f"  - Recall: {best_epoch['metrics/recall(B)']:.4f}")
    report_lines.append("")
    
    report_lines.append("2.3 Métricas Finales (Época {}):".format(total_epochs))
    report_lines.append(f"  - mAP50: {final_metrics['metrics/mAP50(B)']:.4f}")
    report_lines.append(f"  - mAP50-95: {final_metrics['metrics/mAP50-95(B)']:.4f}")
    report_lines.append(f"  - Precision: {final_metrics['metrics/precision(B)']:.4f}")
    report_lines.append(f"  - Recall: {final_metrics['metrics/recall(B)']:.4f}")
    report_lines.append("")
    
    report_lines.append("2.4 Evolución de Pérdidas (Loss):")
    report_lines.append(f"  - Box loss (inicial): {df['train/box_loss'].iloc[0]:.4f}")
    report_lines.append(f"  - Box loss (final): {df['train/box_loss'].iloc[-1]:.4f}")
    report_lines.append(f"  - Cls loss (inicial): {df['train/cls_loss'].iloc[0]:.4f}")
    report_lines.append(f"  - Cls loss (final): {df['train/cls_loss'].iloc[-1]:.4f}")
    report_lines.append(f"  - DFL loss (inicial): {df['train/dfl_loss'].iloc[0]:.4f}")
    report_lines.append(f"  - DFL loss (final): {df['train/dfl_loss'].iloc[-1]:.4f}")
    report_lines.append("")
    
    # Sección 3: Análisis de Convergencia
    report_lines.append("="*80)
    report_lines.append("3. ANÁLISIS DE CONVERGENCIA")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Calcular mejora en pérdidas
    box_improvement = ((df['train/box_loss'].iloc[0] - df['train/box_loss'].iloc[-1]) / 
                       df['train/box_loss'].iloc[0] * 100)
    cls_improvement = ((df['train/cls_loss'].iloc[0] - df['train/cls_loss'].iloc[-1]) / 
                       df['train/cls_loss'].iloc[0] * 100)
    
    report_lines.append("3.1 Reducción de Pérdidas:")
    report_lines.append(f"  - Box loss: {box_improvement:.2f}% de reducción")
    report_lines.append(f"  - Cls loss: {cls_improvement:.2f}% de reducción")
    report_lines.append("")
    
    report_lines.append("3.2 Progreso de mAP:")
    report_lines.append(f"  - mAP50 época 1: {df['metrics/mAP50(B)'].iloc[0]:.4f}")
    report_lines.append(f"  - mAP50 mejor época: {best_epoch['metrics/mAP50(B)']:.4f}")
    report_lines.append(f"  - Mejora total: {(best_epoch['metrics/mAP50(B)'] - df['metrics/mAP50(B)'].iloc[0]):.4f}")
    report_lines.append("")
    
    # Sección 4: Archivos Generados
    report_lines.append("="*80)
    report_lines.append("4. ARCHIVOS DE EVIDENCIA GENERADOS")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("4.1 Pesos del Modelo:")
    report_lines.append(f"  - best.pt: {results_path / 'weights' / 'best.pt'}")
    report_lines.append(f"  - last.pt: {results_path / 'weights' / 'last.pt'}")
    report_lines.append("")
    
    report_lines.append("4.2 Visualizaciones:")
    viz_files = ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                 'F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png',
                 'labels.jpg', 'labels_correlogram.jpg',
                 'train_batch0.jpg', 'train_batch1.jpg', 'train_batch2.jpg']
    
    for viz_file in viz_files:
        if (results_path / viz_file).exists():
            report_lines.append(f"  - {viz_file}")
    report_lines.append("")
    
    report_lines.append("4.3 Datos y Configuración:")
    report_lines.append(f"  - results.csv: Métricas por época")
    report_lines.append(f"  - args.yaml: Configuración completa de entrenamiento")
    report_lines.append("")
    
    # Sección 5: Conclusiones
    report_lines.append("="*80)
    report_lines.append("5. CONCLUSIONES")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Evaluar resultados
    best_map50 = best_epoch['metrics/mAP50(B)']
    
    if best_map50 >= 0.70:
        performance = "EXCELENTE"
    elif best_map50 >= 0.60:
        performance = "BUENO"
    elif best_map50 >= 0.50:
        performance = "ACEPTABLE"
    else:
        performance = "REQUIERE MEJORA"
    
    report_lines.append(f"5.1 Rendimiento General: {performance}")
    report_lines.append(f"  - mAP50 alcanzado: {best_map50:.4f} ({best_map50*100:.2f}%)")
    report_lines.append("")
    
    report_lines.append("5.2 Observaciones:")
    if total_epochs < args.get('epochs', 0):
        report_lines.append(f"  - Early stopping activado en época {total_epochs}")
        report_lines.append(f"    (se detiene {args.get('epochs', 0) - total_epochs} épocas antes del límite)")
    else:
        report_lines.append(f"  - Entrenamiento completó todas las épocas configuradas")
    
    if box_improvement > 50:
        report_lines.append(f"  - Buena convergencia en box loss ({box_improvement:.1f}% reducción)")
    
    if best_epoch['metrics/precision(B)'] > 0.70:
        report_lines.append(f"  - Alta precisión ({best_epoch['metrics/precision(B)']:.4f}), pocas falsas alarmas")
    
    if best_epoch['metrics/recall(B)'] > 0.70:
        report_lines.append(f"  - Alto recall ({best_epoch['metrics/recall(B)']:.4f}), detecta mayoría de objetos")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("FIN DEL REPORTE")
    report_lines.append("="*80)
    
    # Guardar reporte
    report_file = output_path / 'EVIDENCIA_ENTRENAMIENTO.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReporte guardado en: {report_file}")
    
    # 5. Generar resumen JSON para procesamiento
    print("5. Generando resumen JSON...")
    
    summary = {
        'fecha_generacion': datetime.now().isoformat(),
        'configuracion': {
            'modelo': args.get('model'),
            'dataset': args.get('data'),
            'epocas': args.get('epochs'),
            'batch_size': args.get('batch'),
            'img_size': args.get('imgsz'),
            'optimizer': args.get('optimizer'),
            'lr0': args.get('lr0'),
            'device': args.get('device')
        },
        'resultados': {
            'total_epochs': int(total_epochs),
            'mejor_epoca': int(best_epoch['epoch']) + 1,
            'metricas_mejor_modelo': {
                'mAP50': float(best_epoch['metrics/mAP50(B)']),
                'mAP50_95': float(best_epoch['metrics/mAP50-95(B)']),
                'precision': float(best_epoch['metrics/precision(B)']),
                'recall': float(best_epoch['metrics/recall(B)'])
            },
            'metricas_finales': {
                'mAP50': float(final_metrics['metrics/mAP50(B)']),
                'mAP50_95': float(final_metrics['metrics/mAP50-95(B)']),
                'precision': float(final_metrics['metrics/precision(B)']),
                'recall': float(final_metrics['metrics/recall(B)'])
            }
        },
        'archivos': {
            'pesos': str(results_path / 'weights' / 'best.pt'),
            'resultados': str(results_path / 'results.csv'),
            'configuracion': str(results_path / 'args.yaml')
        }
    }
    
    json_file = output_path / 'resumen_entrenamiento.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Resumen JSON guardado en: {json_file}")
    
    # 6. Generar tabla LaTeX para documento académico
    print("6. Generando tabla LaTeX...")
    
    latex_lines = []
    latex_lines.append("% Tabla de resultados de entrenamiento")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Resultados del Entrenamiento - Road Obstacle Detection}")
    latex_lines.append("\\label{tab:training_results}")
    latex_lines.append("\\begin{tabular}{|l|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Métrica} & \\textbf{Mejor Modelo} & \\textbf{Modelo Final} \\\\")
    latex_lines.append("\\hline")
    latex_lines.append(f"Época & {int(best_epoch['epoch']) + 1} & {total_epochs} \\\\")
    latex_lines.append(f"mAP50 & {best_epoch['metrics/mAP50(B)']:.4f} & {final_metrics['metrics/mAP50(B)']:.4f} \\\\")
    latex_lines.append(f"mAP50-95 & {best_epoch['metrics/mAP50-95(B)']:.4f} & {final_metrics['metrics/mAP50-95(B)']:.4f} \\\\")
    latex_lines.append(f"Precision & {best_epoch['metrics/precision(B)']:.4f} & {final_metrics['metrics/precision(B)']:.4f} \\\\")
    latex_lines.append(f"Recall & {best_epoch['metrics/recall(B)']:.4f} & {final_metrics['metrics/recall(B)']:.4f} \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_file = output_path / 'tabla_resultados.tex'
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Tabla LaTeX guardada en: {latex_file}")
    
    # 7. Generar tabla Markdown para README
    print("7. Generando tabla Markdown...")
    
    md_lines = []
    md_lines.append("## Resultados del Entrenamiento")
    md_lines.append("")
    md_lines.append("### Configuración")
    md_lines.append("")
    md_lines.append("| Parámetro | Valor |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| Modelo base | {args.get('model', 'N/A')} |")
    md_lines.append(f"| Épocas | {args.get('epochs', 'N/A')} |")
    md_lines.append(f"| Batch size | {args.get('batch', 'N/A')} |")
    md_lines.append(f"| Imagen size | {args.get('imgsz', 'N/A')} |")
    md_lines.append(f"| Optimizer | {args.get('optimizer', 'N/A')} |")
    md_lines.append(f"| Learning rate | {args.get('lr0', 'N/A')} |")
    md_lines.append("")
    md_lines.append("### Resultados")
    md_lines.append("")
    md_lines.append("| Métrica | Mejor Modelo | Modelo Final |")
    md_lines.append("|---------|--------------|--------------|")
    md_lines.append(f"| Época | {int(best_epoch['epoch']) + 1} | {total_epochs} |")
    md_lines.append(f"| mAP50 | {best_epoch['metrics/mAP50(B)']:.4f} | {final_metrics['metrics/mAP50(B)']:.4f} |")
    md_lines.append(f"| mAP50-95 | {best_epoch['metrics/mAP50-95(B)']:.4f} | {final_metrics['metrics/mAP50-95(B)']:.4f} |")
    md_lines.append(f"| Precision | {best_epoch['metrics/precision(B)']:.4f} | {final_metrics['metrics/precision(B)']:.4f} |")
    md_lines.append(f"| Recall | {best_epoch['metrics/recall(B)']:.4f} | {final_metrics['metrics/recall(B)']:.4f} |")
    md_lines.append("")
    
    md_file = output_path / 'resultados_entrenamiento.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Tabla Markdown guardada en: {md_file}")
    
    # Resumen final
    print("\n" + "="*70)
    print("GENERACIÓN COMPLETADA")
    print("="*70)
    print(f"\nArchivos generados en: {output_path}")
    print(f"  1. EVIDENCIA_ENTRENAMIENTO.txt - Reporte completo")
    print(f"  2. resumen_entrenamiento.json - Datos procesables")
    print(f"  3. tabla_resultados.tex - Para documento LaTeX")
    print(f"  4. resultados_entrenamiento.md - Para README/GitHub")
    print("\nEstos archivos pueden ser usados como evidencia en tu informe académico.")
    print("="*70)

def main():
    """Función principal"""
    print("\nGenerando evidencia del entrenamiento...")
    
    results_path = Path('results/training/road_obstacles_v1')
    output_path = Path('docs/evidencias')
    
    if not results_path.exists():
        print(f"\nERROR: No se encontró el directorio de resultados: {results_path}")
        print("Verifica que el entrenamiento se haya completado correctamente.")
        return
    
    generate_evidence_report(results_path, output_path)

if __name__ == '__main__':
    main()
