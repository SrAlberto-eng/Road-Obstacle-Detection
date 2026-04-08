#!/usr/bin/env python3
"""
Conversor COCO a YOLO
Convierte cualquier dataset en formato COCO a formato YOLO con validación y verificaciones de seguridad.

Características:
- Validación previa del dataset de entrada
- Modo dry-run para previsualizar sin modificar archivos
- Manejo integral de errores y validación
- Estadísticas y reportes detallados

Uso:
    # Prueba primero con dry-run (recomendado)
    python coco_to_yolo.py --input data/animals_coco --output data/animals_yolo --name animals --dry-run
    
    # Conversión real
    python coco_to_yolo.py --input data/animals_coco --output data/animals_yolo --name animals
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime


class CocoToYoloConverter:
    """Conversor COCO a YOLO con validación y verificaciones de seguridad."""
    
    def __init__(self, input_dir: str, output_dir: str, dataset_name: str, dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.dry_run = dry_run
        
        # Estadísticas de validación
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'images_with_annotations': 0,
            'images_without_annotations': 0,
            'invalid_boxes': 0,
            'out_of_bounds_boxes': 0,
            'missing_images': 0,
            'copied_images': 0
        }
        
        # Verificaciones de seguridad
        self.validation_errors = []
        self.validation_warnings = []
    
    def log_error(self, message: str):
        """Registra un mensaje de error."""
        self.validation_errors.append(message)
        print(f"  ERROR: {message}")
    
    def log_warning(self, message: str):
        """Registra un mensaje de advertencia."""
        self.validation_warnings.append(message)
        print(f"  ADVERTENCIA: {message}")
    
    def validate_input_dataset(self) -> bool:
        """
        Valida el dataset de entrada antes de cualquier conversión.
        Retorna True si el dataset es válido, False en caso contrario.
        """
        print(f"\n{'='*70}")
        print(f"Validando dataset de entrada...")
        print(f"{'='*70}")
        
        # Verificar si el directorio de entrada existe
        if not self.input_dir.exists():
            self.log_error(f"El directorio de entrada no existe: {self.input_dir}")
            return False
        
        print(f"Directorio de entrada: {self.input_dir}")
        
        # Detectar splits disponibles
        possible_splits = {
            'train': [
                self.input_dir / 'train' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_train.json',
                self.input_dir / 'train' / 'labels.json'
            ],
            'val': [
                self.input_dir / 'valid' / '_annotations.coco.json',
                self.input_dir / 'val' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_val.json',
                self.input_dir / 'val' / 'labels.json'
            ],
            'test': [
                self.input_dir / 'test' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_test.json',
                self.input_dir / 'test' / 'labels.json'
            ]
        }
        
        detected_splits = {}
        for split, possible_paths in possible_splits.items():
            for path in possible_paths:
                if path.exists():
                    detected_splits[split] = path
                    break
        
        if not detected_splits:
            self.log_error("No se encontraron archivos de anotaciones COCO en ubicaciones esperadas")
            print(f"\nArchivos de anotaciones esperados en una de estas ubicaciones:")
            for split, paths in possible_splits.items():
                print(f"  {split}:")
                for path in paths:
                    print(f"    - {path.relative_to(self.input_dir)}")
            return False
        
        print(f"Splits detectados: {list(detected_splits.keys())}")
        
        # Validar cada split
        total_images = 0
        total_annotations = 0
        
        for split, coco_file in detected_splits.items():
            print(f"\nValidando split {split}...")
            
            # Cargar archivo COCO
            try:
                with open(coco_file, 'r') as f:
                    coco_data = json.load(f)
            except json.JSONDecodeError as e:
                self.log_error(f"{split}: Archivo JSON inválido - {e}")
                return False
            except Exception as e:
                self.log_error(f"{split}: Error al leer archivo - {e}")
                return False
            
            # Validar campos requeridos
            required_keys = ['images', 'annotations', 'categories']
            missing_keys = [k for k in required_keys if k not in coco_data]
            if missing_keys:
                self.log_error(f"{split}: Faltan campos requeridos: {missing_keys}")
                return False
            
            num_images = len(coco_data['images'])
            num_annotations = len(coco_data['annotations'])
            num_categories = len(coco_data['categories'])
            
            if num_images == 0:
                self.log_error(f"{split}: No se encontraron imágenes en archivo COCO")
                return False
            
            if num_categories == 0:
                self.log_error(f"{split}: No se encontraron categorías en archivo COCO")
                return False
            
            print(f"  Imágenes: {num_images}")
            print(f"  Anotaciones: {num_annotations}")
            print(f"  Categorías: {num_categories}")
            
            # Verificar si existen las imágenes
            img_source_dir = coco_file.parent
            sample_images = coco_data['images'][:5]  # Verificar primeras 5 imágenes
            found_count = 0
            
            for img_info in sample_images:
                img_path = self.find_image_file(img_source_dir, img_info['file_name'])
                if img_path:
                    found_count += 1
            
            if found_count == 0:
                self.log_error(f"{split}: No se encontraron archivos de imágenes (verificadas {len(sample_images)} muestras)")
                return False
            elif found_count < len(sample_images):
                self.log_warning(f"{split}: Solo {found_count}/{len(sample_images)} imágenes de muestra encontradas")
            else:
                print(f"  Archivos de imágenes accesibles (verificadas {found_count} muestras)")
            
            total_images += num_images
            total_annotations += num_annotations
        
        print(f"\n{'='*70}")
        print(f"VALIDACIÓN DE ENTRADA EXITOSA")
        print(f"{'='*70}")
        print(f"Total de imágenes: {total_images}")
        print(f"Total de anotaciones: {total_annotations}")
        print(f"Splits: {list(detected_splits.keys())}")
        
        if self.validation_warnings:
            print(f"\nAdvertencias: {len(self.validation_warnings)}")
            for warning in self.validation_warnings[:5]:
                print(f"  - {warning}")
        
        self.detected_splits = detected_splits
        return True
    
    def check_output_directory(self) -> bool:
        """
        Verifica el estado del directorio de salida.
        Retorna True si es seguro continuar, False en caso contrario.
        """
        print(f"\n{'='*70}")
        print(f"VERIFICACIÓN DE SEGURIDAD 2: Directorio de Salida")
        print(f"{'='*70}")
        
        if self.output_dir.exists():
            # Contar archivos existentes
            image_files = list(self.output_dir.rglob('*.jpg')) + list(self.output_dir.rglob('*.png'))
            label_files = list(self.output_dir.rglob('*.txt'))
            
            if image_files or label_files:
                self.log_warning(f"El directorio de salida ya existe con datos:")
                print(f"  Imágenes: {len(image_files)}")
                print(f"  Etiquetas: {len(label_files)}")
                print(f"\n  ¡Estos datos serán SOBRESCRITOS durante la conversión!")
                
                if not self.dry_run:
                    response = input(f"\n  ¿Continuar y sobrescribir? [y/N]: ").strip().lower()
                    if response != 'y':
                        print(f"\n  ERROR: Conversión cancelada por el usuario.")
                        return False
            else:
                print(f"El directorio de salida existe pero está vacío")
        else:
            print(f"El directorio de salida será creado: {self.output_dir}")
        
        return True
    
    def validate_coco_file(self, coco_path: Path) -> Dict:
        """Valida la estructura JSON COCO y la carga."""
        if not coco_path.exists():
            raise FileNotFoundError(f"Archivo COCO no encontrado: {coco_path}")
        
        print(f"  Cargando archivo COCO: {coco_path.name}")
        with open(coco_path, 'r') as f:
            data = json.load(f)
        
        # Validar campos requeridos
        required_keys = ['images', 'annotations', 'categories']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"Faltan campos requeridos en archivo COCO: {missing_keys}")
        
        print(f"    Imágenes: {len(data['images'])}")
        print(f"    Anotaciones: {len(data['annotations'])}")
        print(f"    Categorías: {len(data['categories'])}")
        
        return data
    
    def create_category_mapping(self, categories: List[Dict]) -> Tuple[Dict, List]:
        """Crea el mapeo de ID de categoría a ID de clase YOLO."""
        # Ordenar por ID de categoría para consistencia
        sorted_cats = sorted(categories, key=lambda x: x['id'])
        
        # Crear mapeo: COCO category_id -> YOLO class_id (indexado desde 0)
        cat_to_yolo = {cat['id']: idx for idx, cat in enumerate(sorted_cats)}
        
        # Crear lista de nombres de clases
        class_names = [cat['name'] for cat in sorted_cats]
        
        print(f"\n  Mapeo de Categorías:")
        for cat in sorted_cats:
            yolo_id = cat_to_yolo[cat['id']]
            print(f"    {cat['name']} (COCO ID {cat['id']}) -> YOLO class {yolo_id}")
        
        return cat_to_yolo, class_names
    
    def bbox_coco_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convierte bbox de COCO a formato YOLO con validación.
        COCO: [x_min, y_min, width, height] (píxeles absolutos)
        YOLO: [x_center, y_center, width, height] (normalizado 0-1)
        """
        x_min, y_min, width, height = bbox
        
        # Validar dimensiones de la caja
        if width <= 0 or height <= 0:
            return None  # Caja inválida
        
        # Calcular coordenadas del centro
        x_center = (x_min + width / 2.0) / img_width
        y_center = (y_min + height / 2.0) / img_height
        w_norm = width / img_width
        h_norm = height / img_height
        
        # Validar que las coordenadas normalizadas estén en rango válido [0, 1]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1):
            return None  # Fuera de límites
        
        return x_center, y_center, w_norm, h_norm
    
    def create_yolo_structure(self, splits: List[str]):
        """Crea la estructura de directorios YOLO."""
        print(f"\nCreando estructura de directorios YOLO...")
        
        if self.dry_run:
            print(f"  [DRY RUN] Se crearían los directorios:")
            for split in splits:
                print(f"    - images/{split}/")
                print(f"    - labels/{split}/")
            return
        
        for split in splits:
            img_dir = self.output_dir / 'images' / split
            lbl_dir = self.output_dir / 'labels' / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {split}/")
    
    def find_image_file(self, base_dir: Path, file_name: str) -> Path:
        """Busca el archivo de imagen en base_dir o subdirectorios comunes."""
        # Verificar directamente en base_dir
        img_path = base_dir / file_name
        if img_path.exists():
            return img_path
        
        # Verificar en subdirectorio 'images'
        img_path = base_dir / 'images' / file_name
        if img_path.exists():
            return img_path
        
        # Verificar en otros subdirectorios comunes
        for subdir in ['img', 'imgs', 'data']:
            img_path = base_dir / subdir / file_name
            if img_path.exists():
                return img_path
        
        return None
    
    def convert_split(self, split_name: str, coco_file: Path, img_source_dir: Path):
        """Convierte un split (train/val/test) de COCO a YOLO."""
        print(f"\n{'='*70}")
        print(f"Convirtiendo split: {split_name.upper()}")
        if self.dry_run:
            print(f"[MODO DRY RUN - No se modificarán archivos]")
        print(f"{'='*70}")
        
        # Cargar y validar datos COCO
        coco_data = self.validate_coco_file(coco_file)
        
        # Crear mapeo de categorías
        cat_to_yolo, class_names = self.create_category_mapping(coco_data['categories'])
        
        # Crear mapeo de ID de imagen a info para búsqueda rápida
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        
        # Agrupar anotaciones por ID de imagen
        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)
        
        # Convertir cada imagen
        print(f"\nProcesando imágenes...")
        split_stats = {
            'processed': 0,
            'with_labels': 0,
            'without_labels': 0,
            'invalid_boxes': 0,
            'copied_images': 0,
            'missing_images': 0
        }
        
        for img_id, img_info in img_id_to_info.items():
            file_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Find source image (check multiple locations)
            src_img = self.find_image_file(img_source_dir, file_name)
            
            if src_img is None:
                split_stats['missing_images'] += 1
                if split_stats['missing_images'] <= 3:
                    self.log_warning(f"Image not found: {file_name}")
                continue
            
            # Rutas de destino
            dst_img = self.output_dir / 'images' / split_name / file_name
            dst_lbl = self.output_dir / 'labels' / split_name / (Path(file_name).stem + '.txt')
            
            # DRY RUN: Omitir operaciones reales de archivo
            if not self.dry_run:
                # Copiar imagen
                try:
                    shutil.copy2(src_img, dst_img)
                    split_stats['copied_images'] += 1
                except Exception as e:
                    self.log_error(f"Failed to copy {file_name}: {e}")
                    continue
            else:
                split_stats['copied_images'] += 1
            
            # Convertir anotaciones
            yolo_lines = []
            annotations = img_id_to_anns.get(img_id, [])
            
            for ann in annotations:
                # Omitir anotaciones de multitud
                if ann.get('iscrowd', 0) == 1:
                    continue
                
                # Omitir si no hay bbox
                if 'bbox' not in ann or not ann['bbox']:
                    continue
                
                # Obtener ID de clase YOLO
                coco_cat_id = ann['category_id']
                if coco_cat_id not in cat_to_yolo:
                    if split_stats['invalid_boxes'] == 0:  # Registrar solo una vez
                        self.log_warning(f"ID de categoría desconocido {coco_cat_id}")
                    split_stats['invalid_boxes'] += 1
                    continue
                
                yolo_class_id = cat_to_yolo[coco_cat_id]
                
                # Convertir bbox
                yolo_bbox = self.bbox_coco_to_yolo(ann['bbox'], img_width, img_height)
                
                if yolo_bbox is None:
                    split_stats['invalid_boxes'] += 1
                    continue
                
                x_c, y_c, w, h = yolo_bbox
                yolo_lines.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            
            # DRY RUN: Omitir escritura real de archivos
            if not self.dry_run:
                # Escribir archivo de etiquetas (incluso si está vacío)
                try:
                    with open(dst_lbl, 'w') as f:
                        if yolo_lines:
                            f.write('\n'.join(yolo_lines))
                            split_stats['with_labels'] += 1
                        else:
                            split_stats['without_labels'] += 1
                except Exception as e:
                    self.log_error(f"Error al escribir etiqueta para {file_name}: {e}")
                    continue
            else:
                if yolo_lines:
                    split_stats['with_labels'] += 1
                else:
                    split_stats['without_labels'] += 1
            
            split_stats['processed'] += 1
        
        # Imprimir estadísticas del split
        print(f"\nEstadísticas de {split_name.upper()}:")
        print(f"  Imágenes procesadas: {split_stats['processed']}")
        print(f"  Imágenes copiadas: {split_stats['copied_images']}")
        print(f"  Imágenes con etiquetas: {split_stats['with_labels']}")
        print(f"  Imágenes sin etiquetas: {split_stats['without_labels']}")
        if split_stats['missing_images'] > 0:
            print(f"  ADVERTENCIA: Imágenes no encontradas: {split_stats['missing_images']}")
        if split_stats['invalid_boxes'] > 0:
            print(f"  ADVERTENCIA: Cajas inválidas omitidas: {split_stats['invalid_boxes']}")
        
        # Actualizar estadísticas globales
        self.stats['total_images'] += split_stats['processed']
        self.stats['copied_images'] += split_stats['copied_images']
        self.stats['missing_images'] += split_stats['missing_images']
        self.stats['invalid_boxes'] += split_stats['invalid_boxes']
        
        return class_names
    
    def create_data_yaml(self, class_names: List[str], splits: List[str]):
        """Crea el archivo de configuración data.yaml para YOLO."""
        yaml_path = self.output_dir / 'data.yaml'
        
        yaml_content = f"""# Configuración del Dataset YOLO
# Generado desde formato COCO: {self.dataset_name}
# Fecha de conversión: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {self.output_dir.absolute()}  # directorio raíz del dataset
train: images/train  # imágenes de entrenamiento (relativo a 'path')
val: images/val  # imágenes de validación (relativo a 'path')
"""
        
        if 'test' in splits:
            yaml_content += "test: images/test  # imágenes de prueba (opcional)\n"
        
        yaml_content += f"\n# Clases\nnames:\n"
        for idx, name in enumerate(class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        if self.dry_run:
            print(f"\n[DRY RUN] Se crearía data.yaml:")
            print(f"  Ubicación: {yaml_path}")
            print(f"  Clases: {len(class_names)}")
            return
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nArchivo data.yaml creado con {len(class_names)} clases")
        print(f"  Ubicación: {yaml_path}")
    
    def detect_splits_and_convert(self):
        """Auto-detecta los splits disponibles y los convierte con verificaciones de seguridad."""
        print(f"\n{'='*70}")
        print(f"Conversión COCO a YOLO: {self.dataset_name}")
        if self.dry_run:
            print(f"[MODO DRY RUN - No se modificarán archivos]")
        print(f"{'='*70}")
        print(f"Entrada:  {self.input_dir}")
        print(f"Salida: {self.output_dir}")
        
        # VERIFICACIÓN DE SEGURIDAD 1: Validar dataset de entrada
        if not self.validate_input_dataset():
            print(f"\n{'='*70}")
            print(f"ERROR: VALIDACIÓN DE ENTRADA FALLIDA")
            print(f"{'='*70}")
            print(f"Errores: {len(self.validation_errors)}")
            for error in self.validation_errors:
                print(f"  - {error}")
            return False
        
        # VERIFICACIÓN DE SEGURIDAD 2: Verificar directorio de salida
        if not self.check_output_directory():
            return False
        
        # Crear estructura YOLO
        self.create_yolo_structure(list(self.detected_splits.keys()))
        
        # Convertir cada split
        class_names = None
        for split, coco_file in self.detected_splits.items():
            # Determinar directorio de origen de imágenes
            img_source_dir = coco_file.parent
            
            split_class_names = self.convert_split(split, coco_file, img_source_dir)
            
            # Verificar que todos los splits tengan las mismas clases
            if class_names is None:
                class_names = split_class_names
            elif class_names != split_class_names:
                self.log_warning(f"¡Los nombres de clases difieren entre splits!")
                print(f"  Clases del primer split: {class_names}")
                print(f"  Clases de {split}: {split_class_names}")
        
        # Crear data.yaml
        self.create_data_yaml(class_names, list(self.detected_splits.keys()))
        
        # Reporte final
        print(f"\n{'='*70}")
        if self.dry_run:
            print(f"DRY RUN COMPLETADO - No se modificaron archivos")
        else:
            print(f"CONVERSIÓN COMPLETADA EXITOSAMENTE")
        print(f"{'='*70}")
        
        print(f"\nEstadísticas Finales:")
        print(f"  Imágenes procesadas: {self.stats['total_images']}")
        print(f"  Imágenes copiadas: {self.stats['copied_images']}")
        print(f"  Imágenes faltantes: {self.stats['missing_images']}")
        print(f"  Cajas inválidas: {self.stats['invalid_boxes']}")
        
        if self.validation_warnings:
            print(f"\nADVERTENCIA: Advertencias: {len(self.validation_warnings)}")
            for warning in self.validation_warnings[:10]:
                print(f"  - {warning}")
            if len(self.validation_warnings) > 10:
                print(f"  ... y {len(self.validation_warnings) - 10} más")
        
        if self.validation_errors:
            print(f"\nERROR: Errores encontrados: {len(self.validation_errors)}")
            for error in self.validation_errors[:10]:
                print(f"  - {error}")
            return False
        
        if not self.dry_run:
            print(f"\nDirectorio de salida: {self.output_dir}")
            print(f"\nAhora puedes:")
            print(f"  1. Validar: python scripts/analysis/validate_yolo_dataset.py {self.output_dir}")
            print(f"  2. Entrenar: yolo train data={self.output_dir / 'data.yaml'} model=yolov8n.pt epochs=100")
        else:
            print(f"\nPara realizar la conversión real, ejecuta sin el flag --dry-run")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convierte dataset de formato COCO a formato YOLO con verificaciones de seguridad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Dry run primero (recomendado)
  python coco_to_yolo.py --input data/animals_coco --output data/animals_yolo --name animals --dry-run
  
  # Conversión real
  python coco_to_yolo.py --input data/animals_coco --output data/animals_yolo --name animals
  
  # Convertir múltiples datasets
  python coco_to_yolo.py --input data/pothole_coco --output data/pothole_yolo --name pothole
  python coco_to_yolo.py --input data/road_debris_coco --output data/road_debris_yolo --name road_debris
  python coco_to_yolo.py --input data/self_driving_coco --output data/self_driving_yolo --name self_driving

Características de Seguridad:
  - Validación previa del dataset de entrada
  - Modo dry-run para previsualizar la conversión
  - Manejo integral de errores
  - Estadísticas y reportes detallados
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Directorio del dataset COCO de entrada')
    parser.add_argument('--output', '-o', required=True, help='Directorio del dataset YOLO de salida')
    parser.add_argument('--name', '-n', required=True, help='Nombre del dataset')
    parser.add_argument('--dry-run', action='store_true', help='Previsualizar conversión sin modificar archivos')
    
    args = parser.parse_args()
    
    try:
        converter = CocoToYoloConverter(args.input, args.output, args.name, args.dry_run)
        success = converter.detect_splits_and_convert()
        
        if not success:
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\nERROR: Conversión cancelada por el usuario")
        return 1
    except Exception as e:
        print(f"\nERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
