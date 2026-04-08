"""
Script para rebalancear los splits del dataset
Asegura que train, val y test tengan todas las clases representadas

IMPORTANTE: Este script requiere que el dataset esté completo antes de ejecutarse.
Crea un backup automático antes de cualquier modificación.
"""

import yaml
import shutil
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
import os

def load_dataset_info(yaml_path):
    """Cargar información del dataset"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def get_image_label_pairs(images_dir, labels_dir):
    """Obtener pares imagen-etiqueta con información de clases"""
    pairs = []
    
    for img_path in images_dir.glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
            
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue
        
        # Leer clases en la etiqueta
        classes = set()
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
        
        pairs.append({
            'image': img_path,
            'label': label_path,
            'classes': classes
        })
    
    return pairs

def analyze_current_splits(dataset_path):
    """Analizar distribución actual de los splits"""
    base_path = Path(dataset_path)
    
    print("=" * 60)
    print("ANÁLISIS DE SPLITS ACTUALES")
    print("=" * 60)
    
    splits_info = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split
        
        pairs = get_image_label_pairs(images_dir, labels_dir)
        
        # Contar clases
        class_counts = defaultdict(int)
        for pair in pairs:
            for cls in pair['classes']:
                class_counts[cls] += 1
        
        splits_info[split] = {
            'pairs': pairs,
            'class_counts': dict(class_counts),
            'total': len(pairs)
        }
        
        print(f"\n{split.upper()}:")
        print(f"  Imágenes: {len(pairs)}")
        print(f"  Clases presentes: {sorted(class_counts.keys())}")
        print(f"  Distribución:")
        for cls in range(6):
            count = class_counts.get(cls, 0)
            print(f"    Clase {cls}: {count} imágenes")
    
    return splits_info

def stratified_split(all_pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Crear splits estratificados asegurando que todas las clases estén en todos los splits
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Agrupar por combinación de clases para mantener distribución
    class_groups = defaultdict(list)
    
    for pair in all_pairs:
        # Usar tuple de clases como key
        key = tuple(sorted(pair['classes']))
        class_groups[key].append(pair)
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    # Para cada grupo, dividir estratificadamente
    for key, pairs in class_groups.items():
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_pairs.extend(pairs[:train_end])
        val_pairs.extend(pairs[train_end:val_end])
        test_pairs.extend(pairs[val_end:])
    
    # Shuffle final
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)
    
    return train_pairs, val_pairs, test_pairs

def ensure_all_classes_in_split(pairs, min_per_class=10):
    """Verificar que todas las clases estén representadas"""
    class_counts = defaultdict(int)
    
    for pair in pairs:
        for cls in pair['classes']:
            class_counts[cls] += 1
    
    missing_classes = []
    low_count_classes = []
    
    for cls in range(6):
        count = class_counts.get(cls, 0)
        if count == 0:
            missing_classes.append(cls)
        elif count < min_per_class:
            low_count_classes.append((cls, count))
    
    return missing_classes, low_count_classes, class_counts

def rebalance_dataset(dataset_path, backup=True):
    """
    Rebalancear el dataset completo
    """
    base_path = Path(dataset_path)
    
    print("\n" + "=" * 60)
    print("REBALANCEANDO DATASET")
    print("=" * 60)
    
    # Verificar que el dataset no esté vacío
    print("\nVerificando dataset...")
    total_count = 0
    for split in ['train', 'val', 'test']:
        images_dir = base_path / 'images' / split
        if images_dir.exists():
            count = len(list(images_dir.glob('*.*')))
            total_count += count
            print(f"  {split}: {count} imágenes")
    
    if total_count == 0:
        print("\n❌ ERROR: El dataset está vacío!")
        print("   Por favor, restaura el dataset antes de rebalancear.")
        return False
    
    # Backup (crear copia de seguridad COMPLETA primero)
    if backup:
        backup_path = base_path.parent / f"{base_path.name}_backup"
        
        if backup_path.exists():
            print(f"\n⚠️  Backup ya existe: {backup_path}")
            response = input("¿Sobrescribir backup? (y/n): ")
            if response.lower() != 'y':
                print("Cancelado.")
                return False
            shutil.rmtree(backup_path)
        
        print(f"\nCreando backup en: {backup_path}")
        shutil.copytree(base_path, backup_path)
        print("✓ Backup completado")
    
    # Recolectar TODOS los pares imagen-etiqueta desde el BACKUP
    print("\nRecolectando todas las imágenes desde el backup...")
    all_pairs = []
    backup_path = base_path.parent / f"{base_path.name}_backup"
    
    for split in ['train', 'val', 'test']:
        images_dir = backup_path / 'images' / split
        labels_dir = backup_path / 'labels' / split
        
        pairs = get_image_label_pairs(images_dir, labels_dir)
        all_pairs.extend(pairs)
    
    print(f"Total de pares recolectados: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        print("\n❌ ERROR: No se encontraron imágenes con etiquetas!")
        print("   Verifica que el dataset esté completo.")
        return False
    
    # Análisis de clases globales
    print("\nDistribución global de clases:")
    global_class_counts = defaultdict(int)
    for pair in all_pairs:
        for cls in pair['classes']:
            global_class_counts[cls] += 1
    
    for cls in range(6):
        count = global_class_counts.get(cls, 0)
        print(f"  Clase {cls}: {count} imágenes")
    
    # Crear nuevos splits estratificados
    print("\nCreando nuevos splits estratificados...")
    train_pairs, val_pairs, test_pairs = stratified_split(
        all_pairs,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print(f"\nNuevos splits:")
    print(f"  Train: {len(train_pairs)} imágenes ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} imágenes ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} imágenes ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    
    # Verificar que todas las clases están en todos los splits
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE SPLITS")
    print("=" * 60)
    
    for split_name, split_pairs in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
        missing, low, counts = ensure_all_classes_in_split(split_pairs, min_per_class=5)
        
        print(f"\n{split_name}:")
        for cls in range(6):
            count = counts.get(cls, 0)
            status = "✓" if count >= 5 else "⚠️"
            print(f"  {status} Clase {cls}: {count} imágenes")
        
        if missing:
            print(f"  ❌ CLASES FALTANTES: {missing}")
        if low:
            print(f"  ⚠️  Clases con pocas muestras: {low}")
    
    # Limpiar directorios actuales
    print("\n" + "=" * 60)
    print("MOVIENDO ARCHIVOS A NUEVOS SPLITS")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        images_dir = base_path / 'images' / split
        labels_dir = base_path / 'labels' / split
        
        # Limpiar
        for f in images_dir.glob('*.*'):
            f.unlink()
        for f in labels_dir.glob('*.txt'):
            f.unlink()
    
    # Copiar a nuevos splits desde el backup
    backup_path = base_path.parent / f"{base_path.name}_backup"
    
    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        images_dir = base_path / 'images' / split_name
        labels_dir = base_path / 'labels' / split_name
        
        print(f"\nCopiando a {split_name}...")
        for i, pair in enumerate(split_pairs, 1):
            # Las rutas en pair apuntan a archivos que ya no existen
            # Construir ruta desde backup
            img_name = pair['image'].name
            label_name = pair['label'].name
            
            # Buscar en todos los splits del backup
            source_img = None
            source_label = None
            
            for old_split in ['train', 'val', 'test']:
                candidate_img = backup_path / 'images' / old_split / img_name
                candidate_label = backup_path / 'labels' / old_split / label_name
                
                if candidate_img.exists() and candidate_label.exists():
                    source_img = candidate_img
                    source_label = candidate_label
                    break
            
            if source_img and source_label:
                # Copiar imagen
                dest_img = images_dir / img_name
                shutil.copy2(source_img, dest_img)
                
                # Copiar etiqueta
                dest_label = labels_dir / label_name
                shutil.copy2(source_label, dest_label)
            else:
                print(f"  ⚠️  No se encontró: {img_name}")
            
            if i % 500 == 0:
                print(f"  Copiado {i}/{len(split_pairs)}...")
        
        print(f"  ✓ Completado: {len(split_pairs)} pares")
    
    print("\n" + "=" * 60)
    print("✅ REBALANCEO COMPLETADO")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("  1. Ejecutar: python scripts/analysis/validate_yolo_dataset.py --dataset data/road_obstacles_final")
    print("  2. Ejecutar: python scripts/analysis/analyze_dataset_balance.py --dataset data/road_obstacles_final")
    print("  3. Si todo está bien, entrenar: python train.py")
    print(f"\nBackup guardado en: {backup_path}")
    
    return True

def main():
    dataset_path = 'data/road_obstacles_final'
    
    print("\n" + "=" * 60)
    print("REBALANCEADOR DE SPLITS - ROAD OBSTACLE DETECTION")
    print("=" * 60)
    print("\n⚠️  ADVERTENCIA:")
    print("  Este script redistribuirá las imágenes entre train/val/test")
    print("  Se creará un backup automático antes de cualquier cambio")
    print("  Asegúrate de tener espacio suficiente en disco")
    
    # Analizar estado actual
    splits_info = analyze_current_splits(dataset_path)
    
    # Verificar que hay datos
    total_images = sum(info['total'] for info in splits_info.values())
    if total_images == 0:
        print("\n" + "=" * 60)
        print("❌ ERROR: NO HAY IMÁGENES EN EL DATASET")
        print("=" * 60)
        print("\nPor favor:")
        print("  1. Descarga los datasets originales")
        print("  2. Ejecuta: python scripts/merge_road_datasets.py")
        print("  3. Luego vuelve a ejecutar este script")
        return
    
    # Verificar problemas
    print("\n" + "=" * 60)
    print("PROBLEMAS DETECTADOS")
    print("=" * 60)
    
    problems = []
    
    for split_name, info in splits_info.items():
        missing_classes = [c for c in range(6) if c not in info['class_counts']]
        if missing_classes:
            problems.append(f"{split_name}: Clases faltantes {missing_classes}")
            print(f"❌ {split_name}: Clases faltantes {missing_classes}")
    
    if not problems:
        print("✓ No se detectaron problemas de clases faltantes")
        print("\nEl dataset parece estar bien balanceado.")
        print("Si aún deseas rebalancear, continúa.")
    
    # Preguntar si rebalancear
    print("\n" + "=" * 60)
    print("¿REBALANCEAR DATASET?")
    print("=" * 60)
    print("\nEsto hará:")
    print("  1. Backup del dataset actual")
    print("  2. Redistribuir todas las imágenes en splits 70/15/15")
    print("  3. Asegurar que todas las clases estén en todos los splits")
    print("  4. Mantener distribución estratificada")
    
    response = input("\n¿Continuar? (y/n): ")
    
    if response.lower() == 'y':
        success = rebalance_dataset(dataset_path, backup=True)
        if success:
            print("\n✅ Proceso completado exitosamente")
        else:
            print("\n❌ Proceso cancelado o falló")
    else:
        print("\nCancelado.")

if __name__ == '__main__':
    main()
