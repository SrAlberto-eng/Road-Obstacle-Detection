"""
Script para extraer épocas específicas del results.csv y generar tabla formateada.
Útil para incluir en informes académicos.
"""

import pandas as pd
from pathlib import Path

def extract_specific_epochs(csv_path, output_path, epochs_to_extract=None):
    """
    Extrae épocas específicas del results.csv y genera tablas formateadas.
    
    Args:
        csv_path: Path al archivo results.csv
        output_path: Path donde guardar las tablas
        epochs_to_extract: Lista de épocas a extraer (None = automático)
    """
    print("="*70)
    print("EXTRACCIÓN DE ÉPOCAS ESPECÍFICAS PARA REPORTE")
    print("="*70)
    
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"\nERROR: No se encontró {csv_path}")
        return
    
    # Leer CSV
    print(f"\n1. Leyendo {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    total_epochs = len(df)
    print(f"   Total de épocas: {total_epochs}")
    
    # Determinar épocas a extraer
    if epochs_to_extract is None:
        # Épocas automáticas: inicio, cuartiles, mejor, final
        best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
        best_epoch_num = int(df.loc[best_epoch_idx, 'epoch']) + 1
        
        epochs_to_extract = [
            1,                          # Primera época
            total_epochs // 4,          # 25% progreso
            total_epochs // 2,          # 50% progreso
            3 * total_epochs // 4,      # 75% progreso
            best_epoch_num,             # Mejor modelo
            total_epochs                # Última época
        ]
        epochs_to_extract = sorted(set(epochs_to_extract))  # Remover duplicados
    
    print(f"   Épocas a extraer: {epochs_to_extract}")
    
    # Extraer datos
    print("\n2. Extrayendo datos...")
    
    selected_rows = []
    for epoch_num in epochs_to_extract:
        # epoch en CSV es 0-indexed, epoch_num es 1-indexed
        row_idx = epoch_num - 1
        if 0 <= row_idx < len(df):
            row = df.iloc[row_idx].copy()
            row['epoch_display'] = epoch_num  # Para mostrar
            selected_rows.append(row)
    
    if not selected_rows:
        print("   ERROR: No se encontraron épocas válidas")
        return
    
    selected_df = pd.DataFrame(selected_rows)
    
    # Identificar mejor época
    best_epoch_idx_in_selection = selected_df['metrics/mAP50(B)'].idxmax()
    best_epoch_num_in_selection = int(selected_df.loc[best_epoch_idx_in_selection, 'epoch_display'])
    
    # 3. Generar tabla Markdown
    print("\n3. Generando tabla Markdown...")
    
    md_lines = []
    md_lines.append("## Evolución de Métricas Durante el Entrenamiento")
    md_lines.append("")
    md_lines.append("| Época | Box Loss | Cls Loss | DFL Loss | mAP50  | mAP50-95 | Precision | Recall |")
    md_lines.append("|-------|----------|----------|----------|--------|----------|-----------|--------|")
    
    for _, row in selected_df.iterrows():
        epoch_display = int(row['epoch_display'])
        marker = "*" if epoch_display == best_epoch_num_in_selection else " "
        
        md_lines.append(
            f"| {epoch_display:>3}{marker}  | "
            f"{row['train/box_loss']:.4f}   | "
            f"{row['train/cls_loss']:.4f}   | "
            f"{row['train/dfl_loss']:.4f}   | "
            f"{row['metrics/mAP50(B)']:.4f} | "
            f"{row['metrics/mAP50-95(B)']:.4f} | "
            f"{row['metrics/precision(B)']:.4f}    | "
            f"{row['metrics/recall(B)']:.4f}   |"
        )
    
    md_lines.append("")
    md_lines.append(f"\\* Mejor modelo (época {best_epoch_num_in_selection})")
    
    md_file = output_path / 'epocas_seleccionadas.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"   Guardado en: {md_file}")
    
    # 4. Generar tabla LaTeX
    print("\n4. Generando tabla LaTeX...")
    
    latex_lines = []
    latex_lines.append("% Tabla de evolución de métricas durante entrenamiento")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Evolución de Métricas Durante el Entrenamiento}")
    latex_lines.append("\\label{tab:training_evolution}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{|c|c|c|c|c|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Época} & \\textbf{Box Loss} & \\textbf{Cls Loss} & \\textbf{DFL Loss} & "
                      "\\textbf{mAP50} & \\textbf{mAP50-95} & \\textbf{Precision} & \\textbf{Recall} \\\\")
    latex_lines.append("\\hline")
    
    for _, row in selected_df.iterrows():
        epoch_display = int(row['epoch_display'])
        marker = "$^*$" if epoch_display == best_epoch_num_in_selection else ""
        
        latex_lines.append(
            f"{epoch_display}{marker} & "
            f"{row['train/box_loss']:.4f} & "
            f"{row['train/cls_loss']:.4f} & "
            f"{row['train/dfl_loss']:.4f} & "
            f"{row['metrics/mAP50(B)']:.4f} & "
            f"{row['metrics/mAP50-95(B)']:.4f} & "
            f"{row['metrics/precision(B)']:.4f} & "
            f"{row['metrics/recall(B)']:.4f} \\\\"
        )
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append(f"\\\\[5pt] $^*$ Mejor modelo (época {best_epoch_num_in_selection})")
    latex_lines.append("\\end{table}")
    
    latex_file = output_path / 'epocas_seleccionadas.tex'
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"   Guardado en: {latex_file}")
    
    # 5. Generar tabla de texto plano (para copiar/pegar)
    print("\n5. Generando tabla de texto plano...")
    
    txt_lines = []
    txt_lines.append("="*100)
    txt_lines.append("EVOLUCIÓN DE MÉTRICAS DURANTE EL ENTRENAMIENTO")
    txt_lines.append("="*100)
    txt_lines.append("")
    txt_lines.append(f"{'Época':>6} | {'Box Loss':>9} | {'Cls Loss':>9} | {'DFL Loss':>9} | "
                    f"{'mAP50':>7} | {'mAP50-95':>8} | {'Precision':>9} | {'Recall':>7}")
    txt_lines.append("-"*100)
    
    for _, row in selected_df.iterrows():
        epoch_display = int(row['epoch_display'])
        marker = "*" if epoch_display == best_epoch_num_in_selection else " "
        
        txt_lines.append(
            f"{epoch_display:>5}{marker} | "
            f"{row['train/box_loss']:>9.4f} | "
            f"{row['train/cls_loss']:>9.4f} | "
            f"{row['train/dfl_loss']:>9.4f} | "
            f"{row['metrics/mAP50(B)']:>7.4f} | "
            f"{row['metrics/mAP50-95(B)']:>8.4f} | "
            f"{row['metrics/precision(B)']:>9.4f} | "
            f"{row['metrics/recall(B)']:>7.4f}"
        )
    
    txt_lines.append("-"*100)
    txt_lines.append(f"* Mejor modelo (época {best_epoch_num_in_selection})")
    txt_lines.append("")
    
    txt_file = output_path / 'epocas_seleccionadas.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))
    
    print(f"   Guardado en: {txt_file}")
    
    # 6. Generar análisis estadístico
    print("\n6. Generando análisis estadístico...")
    
    analysis_lines = []
    analysis_lines.append("="*70)
    analysis_lines.append("ANÁLISIS ESTADÍSTICO DEL ENTRENAMIENTO")
    analysis_lines.append("="*70)
    analysis_lines.append("")
    
    # Calcular cambios porcentuales
    first_row = selected_df.iloc[0]
    last_row = selected_df.iloc[-1]
    
    box_loss_change = ((last_row['train/box_loss'] - first_row['train/box_loss']) / 
                       first_row['train/box_loss'] * 100)
    cls_loss_change = ((last_row['train/cls_loss'] - first_row['train/cls_loss']) / 
                       first_row['train/cls_loss'] * 100)
    dfl_loss_change = ((last_row['train/dfl_loss'] - first_row['train/dfl_loss']) / 
                       first_row['train/dfl_loss'] * 100)
    map50_change = last_row['metrics/mAP50(B)'] - first_row['metrics/mAP50(B)']
    
    analysis_lines.append("1. CAMBIOS ENTRE PRIMERA Y ÚLTIMA ÉPOCA:")
    analysis_lines.append(f"   Box Loss:  {first_row['train/box_loss']:.4f} → {last_row['train/box_loss']:.4f} "
                         f"({box_loss_change:+.2f}%)")
    analysis_lines.append(f"   Cls Loss:  {first_row['train/cls_loss']:.4f} → {last_row['train/cls_loss']:.4f} "
                         f"({cls_loss_change:+.2f}%)")
    analysis_lines.append(f"   DFL Loss:  {first_row['train/dfl_loss']:.4f} → {last_row['train/dfl_loss']:.4f} "
                         f"({dfl_loss_change:+.2f}%)")
    analysis_lines.append(f"   mAP50:     {first_row['metrics/mAP50(B)']:.4f} → {last_row['metrics/mAP50(B)']:.4f} "
                         f"({map50_change:+.4f})")
    analysis_lines.append("")
    
    # Mejor época
    best_row = selected_df.loc[best_epoch_idx_in_selection]
    analysis_lines.append("2. MEJOR MODELO:")
    analysis_lines.append(f"   Época:     {best_epoch_num_in_selection}")
    analysis_lines.append(f"   mAP50:     {best_row['metrics/mAP50(B)']:.4f}")
    analysis_lines.append(f"   mAP50-95:  {best_row['metrics/mAP50-95(B)']:.4f}")
    analysis_lines.append(f"   Precision: {best_row['metrics/precision(B)']:.4f}")
    analysis_lines.append(f"   Recall:    {best_row['metrics/recall(B)']:.4f}")
    analysis_lines.append("")
    
    # Interpretación
    analysis_lines.append("3. INTERPRETACIÓN:")
    
    if box_loss_change < -20:
        analysis_lines.append("   [+] Box loss muestra excelente convergencia (>20% reducción)")
    elif box_loss_change < -10:
        analysis_lines.append("   [+] Box loss muestra buena convergencia (>10% reducción)")
    else:
        analysis_lines.append("   [!] Box loss muestra convergencia limitada (<10% reducción)")
    
    if cls_loss_change < -30:
        analysis_lines.append("   [+] Cls loss muestra excelente convergencia (>30% reducción)")
    elif cls_loss_change < -15:
        analysis_lines.append("   [+] Cls loss muestra buena convergencia (>15% reducción)")
    else:
        analysis_lines.append("   [!] Cls loss muestra convergencia limitada (<15% reducción)")
    
    if best_row['metrics/precision(B)'] > 0.70:
        analysis_lines.append("   [+] Alta precision (>0.70), pocas falsas alarmas")
    elif best_row['metrics/precision(B)'] > 0.60:
        analysis_lines.append("   [~] Precision moderada (0.60-0.70)")
    else:
        analysis_lines.append("   [-] Precision baja (<0.60), muchas falsas alarmas")
    
    if best_row['metrics/recall(B)'] > 0.70:
        analysis_lines.append("   [+] Alto recall (>0.70), detecta mayoría de objetos")
    elif best_row['metrics/recall(B)'] > 0.50:
        analysis_lines.append("   [~] Recall moderado (0.50-0.70), detecta algunos objetos")
    else:
        analysis_lines.append("   [-] Recall bajo (<0.50), pierde muchos objetos")
    
    analysis_lines.append("")
    analysis_lines.append("="*70)
    
    analysis_file = output_path / 'analisis_estadistico.txt'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis_lines))
    
    print(f"   Guardado en: {analysis_file}")
    
    # Resumen
    print("\n" + "="*70)
    print("GENERACIÓN COMPLETADA")
    print("="*70)
    print(f"\nArchivos generados en: {output_path}")
    print("  1. epocas_seleccionadas.txt - Tabla de texto plano")
    print("  2. epocas_seleccionadas.md - Tabla Markdown")
    print("  3. epocas_seleccionadas.tex - Tabla LaTeX")
    print("  4. analisis_estadistico.txt - Análisis e interpretación")
    print("\nÉpocas extraídas:", epochs_to_extract)
    print(f"Mejor época: {best_epoch_num_in_selection} (mAP50: {best_row['metrics/mAP50(B)']:.4f})")
    print("="*70)

def main():
    """Función principal"""
    print("\nExtrayendo épocas específicas para reporte académico...")
    
    csv_path = Path('results/training/road_obstacles_v1/results.csv')
    output_path = Path('docs/evidencias')
    
    if not csv_path.exists():
        print(f"\nERROR: No se encontró {csv_path}")
        print("Verifica que el entrenamiento se haya completado.")
        return
    
    # Puedes especificar épocas manualmente aquí si quieres
    # epochs_to_extract = [1, 25, 50, 75, 100, 114, 140]
    epochs_to_extract = None  # Automático
    
    extract_specific_epochs(csv_path, output_path, epochs_to_extract)

if __name__ == '__main__':
    main()
