"""
Script para crear un dataset balanceado de debris (escombros) y baches.

Este script extrae las imágenes de debris y baches del dataset consolidado
y crea un nuevo dataset con partes iguales de cada clase para entrenamiento.

Uso:
    python scripts/data/crear_dataset_debris_baches.py
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random
import yaml

def contar_clases_en_etiqueta(ruta_etiqueta, mapa_clases):
    """
    Cuenta qué clases están presentes en un archivo de etiqueta.
    
    Args:
        ruta_etiqueta: Path al archivo .txt de etiquetas
        mapa_clases: Diccionario {id_clase: nombre_clase}
        
    Returns:
        set: Conjunto de nombres de clases presentes
    """
    clases_presentes = set()
    
    with open(ruta_etiqueta, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                id_clase = int(linea.split()[0])
                if id_clase in mapa_clases:
                    clases_presentes.add(mapa_clases[id_clase])
    
    return clases_presentes


def recolectar_imagenes_por_clase(dataset_raiz):
    """
    Recolecta todas las imágenes que contienen debris o baches.
    
    Args:
        dataset_raiz: Path al dataset consolidado
        
    Returns:
        dict: {clase: lista de (imagen_path, etiqueta_path)}
    """
    dataset_raiz = Path(dataset_raiz)
    
    # Cargar información del dataset
    archivo_yaml = dataset_raiz / 'data.yaml'
    with open(archivo_yaml, 'r') as f:
        info_datos = yaml.safe_load(f)
    
    nombres_clases = info_datos['names']
    
    # Mapeo de IDs a nombres (buscamos debris y pothole)
    mapa_clases = {}
    id_debris = None
    id_pothole = None
    
    for id_clase, nombre in nombres_clases.items():
        if 'debris' in nombre.lower() or 'escombro' in nombre.lower():
            mapa_clases[id_clase] = 'debris'
            id_debris = id_clase
        elif 'pothole' in nombre.lower() or 'bache' in nombre.lower():
            mapa_clases[id_clase] = 'pothole'
            id_pothole = id_clase
    
    if id_debris is None or id_pothole is None:
        print("Error: No se encontraron las clases 'debris' y 'pothole' en el dataset")
        print(f"Clases disponibles: {nombres_clases}")
        return None
    
    print(f"Clase debris encontrada: ID {id_debris}")
    print(f"Clase pothole encontrada: ID {id_pothole}")
    print()
    
    # Recolectar imágenes por clase
    imagenes_por_clase = defaultdict(list)
    
    # Procesar todos los splits (train, val, test)
    for split in ['train', 'val', 'test']:
        carpeta_imagenes = dataset_raiz / 'images' / split
        carpeta_etiquetas = dataset_raiz / 'labels' / split
        
        if not carpeta_imagenes.exists():
            continue
        
        print(f"Procesando split: {split}")
        
        # Procesar cada imagen
        for archivo_imagen in carpeta_imagenes.glob('*.*'):
            if archivo_imagen.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Buscar archivo de etiqueta correspondiente
            archivo_etiqueta = carpeta_etiquetas / f"{archivo_imagen.stem}.txt"
            
            if not archivo_etiqueta.exists():
                continue
            
            # Verificar qué clases contiene
            clases = contar_clases_en_etiqueta(archivo_etiqueta, mapa_clases)
            
            # Si contiene debris, añadir a lista de debris
            if 'debris' in clases:
                imagenes_por_clase['debris'].append((archivo_imagen, archivo_etiqueta))
            
            # Si contiene pothole, añadir a lista de pothole
            if 'pothole' in clases:
                imagenes_por_clase['pothole'].append((archivo_imagen, archivo_etiqueta))
    
    return imagenes_por_clase, id_debris, id_pothole


def convertir_etiquetas(archivo_origen, archivo_destino, id_antigua, id_nueva):
    """
    Convierte IDs de clase en archivo de etiquetas.
    
    Args:
        archivo_origen: Archivo de etiqueta original
        archivo_destino: Archivo de etiqueta destino
        id_antigua: ID de clase original
        id_nueva: Nueva ID de clase
    """
    with open(archivo_origen, 'r') as f:
        lineas = f.readlines()
    
    with open(archivo_destino, 'w') as f:
        for linea in lineas:
            partes = linea.strip().split()
            if partes:
                id_clase = int(partes[0])
                # Solo mantener la clase que nos interesa, con nuevo ID
                if id_clase == id_antigua:
                    f.write(f"{id_nueva} {' '.join(partes[1:])}\n")


def crear_dataset_balanceado(imagenes_por_clase, ids_clases, carpeta_salida):
    """
    Crea un dataset balanceado con partes iguales de debris y baches.
    
    Args:
        imagenes_por_clase: Dict con listas de imágenes por clase
        ids_clases: Tupla (id_debris, id_pothole)
        carpeta_salida: Path donde crear el nuevo dataset
    """
    carpeta_salida = Path(carpeta_salida)
    id_debris, id_pothole = ids_clases
    
    # Contar imágenes
    n_debris = len(imagenes_por_clase['debris'])
    n_pothole = len(imagenes_por_clase['pothole'])
    
    print(f"\nImágenes encontradas:")
    print(f"  Debris: {n_debris}")
    print(f"  Pothole: {n_pothole}")
    
    # Determinar cantidad para balancear (usar el menor)
    n_por_clase = min(n_debris, n_pothole)
    
    print(f"\nCreando dataset balanceado con {n_por_clase} imágenes por clase")
    print(f"Total: {n_por_clase * 2} imágenes")
    
    # Mezclar y seleccionar
    random.seed(42)  # Para reproducibilidad
    debris_seleccionados = random.sample(imagenes_por_clase['debris'], n_por_clase)
    pothole_seleccionados = random.sample(imagenes_por_clase['pothole'], n_por_clase)
    
    # Combinar y mezclar todas las imágenes
    todas_imagenes = [('debris', img, lbl) for img, lbl in debris_seleccionados] + \
                     [('pothole', img, lbl) for img, lbl in pothole_seleccionados]
    random.shuffle(todas_imagenes)
    
    # Dividir en train/val/test (80%/15%/5%)
    n_total = len(todas_imagenes)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.15)
    
    splits = {
        'train': todas_imagenes[:n_train],
        'val': todas_imagenes[n_train:n_train + n_val],
        'test': todas_imagenes[n_train + n_val:]
    }
    
    print(f"\nDivisión del dataset:")
    print(f"  Train: {len(splits['train'])} imágenes ({len(splits['train'])/n_total*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} imágenes ({len(splits['val'])/n_total*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} imágenes ({len(splits['test'])/n_total*100:.1f}%)")
    
    # Crear estructura de carpetas
    for split in ['train', 'val', 'test']:
        (carpeta_salida / 'images' / split).mkdir(parents=True, exist_ok=True)
        (carpeta_salida / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copiar archivos
    print("\nCopiando archivos...")
    
    for split, imagenes in splits.items():
        print(f"  {split}: ", end='', flush=True)
        
        conteo_clases = defaultdict(int)
        
        for clase, imagen_origen, etiqueta_origen in imagenes:
            # Generar nombre único para evitar colisiones
            nombre_base = f"{clase}_{imagen_origen.stem}"
            extension = imagen_origen.suffix
            
            # Rutas de destino
            imagen_destino = carpeta_salida / 'images' / split / f"{nombre_base}{extension}"
            etiqueta_destino = carpeta_salida / 'labels' / split / f"{nombre_base}.txt"
            
            # Copiar imagen
            shutil.copy2(imagen_origen, imagen_destino)
            
            # Convertir y copiar etiquetas (reasignar IDs: debris=0, pothole=1)
            if clase == 'debris':
                convertir_etiquetas(etiqueta_origen, etiqueta_destino, id_debris, 0)
                conteo_clases['debris'] += 1
            else:
                convertir_etiquetas(etiqueta_origen, etiqueta_destino, id_pothole, 1)
                conteo_clases['pothole'] += 1
        
        print(f"✓ (debris: {conteo_clases['debris']}, pothole: {conteo_clases['pothole']})")
    
    # Crear archivo data.yaml
    yaml_contenido = {
        'path': str(carpeta_salida.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': {
            0: 'debris',
            1: 'pothole'
        }
    }
    
    archivo_yaml = carpeta_salida / 'data.yaml'
    with open(archivo_yaml, 'w') as f:
        yaml.dump(yaml_contenido, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Archivo data.yaml creado: {archivo_yaml}")
    
    # Crear archivo README
    readme_contenido = f"""# Dataset Balanceado: Debris y Baches

## Descripción
Dataset balanceado con partes iguales de escombros (debris) y baches (potholes).

## Estadísticas
- **Total imágenes**: {n_total}
- **Clases**: 2 (debris, pothole)
- **Balanceo**: 50% debris, 50% pothole

## Distribución
- **Train**: {len(splits['train'])} imágenes ({len(splits['train'])/n_total*100:.1f}%)
- **Val**: {len(splits['val'])} imágenes ({len(splits['val'])/n_total*100:.1f}%)
- **Test**: {len(splits['test'])} imágenes ({len(splits['test'])/n_total*100:.1f}%)

## Clases
- **0: debris** - Escombros en la carretera
- **1: pothole** - Baches en el pavimento

## Uso
```bash
python scripts/training/train_binary_finetune.py \\
    --datos data/debris_pothole_balanced/data.yaml \\
    --epocas 100 \\
    --nombre debris_pothole_model
```

## Origen
Creado a partir del dataset consolidado `road_obstacles_final`.
Fecha de creación: {Path(__file__).stat().st_mtime}
"""
    
    archivo_readme = carpeta_salida / 'README.md'
    with open(archivo_readme, 'w', encoding='utf-8') as f:
        f.write(readme_contenido)
    
    print(f"✓ README creado: {archivo_readme}")


def principal():
    """
    Función principal que ejecuta la creación del dataset.
    """
    # Rutas
    dataset_origen = Path('data/road_obstacles_final')
    dataset_destino = Path('data/debris_pothole_balanced')
    
    # Validar que existe el dataset origen
    if not dataset_origen.exists():
        print(f"Error: No se encuentra el dataset origen en {dataset_origen}")
        print("Asegúrate de que el dataset consolidado esté disponible.")
        return
    
    print(f"Dataset origen: {dataset_origen}")
    print(f"Dataset destino: {dataset_destino}")
    print()
    
    # Recolectar imágenes
    print("Paso 1: Recolectando imágenes de debris y baches...")
    resultado = recolectar_imagenes_por_clase(dataset_origen)
    
    if resultado is None:
        return
    
    imagenes_por_clase, id_debris, id_pothole = resultado
    
    if not imagenes_por_clase['debris'] or not imagenes_por_clase['pothole']:
        print("Error: No se encontraron suficientes imágenes de ambas clases")
        return
    
    # Crear dataset balanceado
    print("\nPaso 2: Creando dataset balanceado...")
    crear_dataset_balanceado(
        imagenes_por_clase,
        (id_debris, id_pothole),
        dataset_destino
    )
    

if __name__ == "__main__":
    principal()