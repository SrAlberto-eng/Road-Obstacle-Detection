#!/usr/bin/env python3
"""
Conversor COCO a YOLO
Convierte cualquier dataset en formato COCO a formato YOLO.

Uso:
    python coco_to_yolo.py --input data/animals_coco --output data/animals_yolo --name animals
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime


class CocoToYoloConverter:
    """Conversor COCO a YOLO con validación básica."""
    
    def __init__(self, input_dir: str, output_dir: str, dataset_name: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.stats = {'processed': 0, 'missing': 0, 'invalid_boxes': 0}
    
    def find_coco_files(self) -> Dict[str, Path]:
        """Detecta automáticamente los archivos de anotaciones COCO."""
        # Ubicaciones comunes de archivos COCO
        locations = {
            'train': [
                self.input_dir / 'train' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_train.json',
            ],
            'val': [
                self.input_dir / 'valid' / '_annotations.coco.json',
                self.input_dir / 'val' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_val.json',
            ],
            'test': [
                self.input_dir / 'test' / '_annotations.coco.json',
                self.input_dir / 'annotations' / 'instances_test.json',
            ]
        }
        
        found = {}
        for split, paths in locations.items():
            for path in paths:
                if path.exists():
                    found[split] = path
                    break
        
        if not found:
            raise FileNotFoundError("No se encontraron archivos de anotaciones COCO")
        
        return found
    
    def find_image(self, base_dir: Path, filename: str) -> Path:
        """Busca una imagen en ubicaciones comunes."""
        for subdir in [base_dir, base_dir / 'images', base_dir / 'img']:
            path = subdir / filename
            if path.exists():
                return path
        return None
    
    def bbox_coco_to_yolo(self, bbox: List[float], img_w: int, img_h: int) -> Tuple:
        """Convierte bbox de COCO [x,y,w,h] a YOLO [x_center,y_center,w,h] normalizado."""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Validar rango [0,1]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1):
            return None
        
        return x_center, y_center, w_norm, h_norm
    
    def convert_split(self, split: str, coco_file: Path):
        """Convierte un split de COCO a YOLO."""
        print(f"\n{'='*60}")
        print(f"Convirtiendo: {split}")
        print(f"{'='*60}")
        
        # Cargar COCO
        with open(coco_file, 'r') as f:
            data = json.load(f)
        
        images = {img['id']: img for img in data['images']}
        categories = sorted(data['categories'], key=lambda x: x['id'])
        cat_map = {cat['id']: idx for idx, cat in enumerate(categories)}
        class_names = [cat['name'] for cat in categories]
        
        print(f"Imágenes: {len(images)}, Clases: {len(class_names)}")
        
        # Agrupar anotaciones por imagen
        anns_by_img = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        
        # Crear directorios
        img_dir = self.output_dir / 'images' / split
        lbl_dir = self.output_dir / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Procesar imágenes
        img_source = coco_file.parent
        for img_id, img_info in images.items():
            filename = img_info['file_name']
            src = self.find_image(img_source, filename)
            
            if not src:
                self.stats['missing'] += 1
                continue
            
            # Copiar imagen
            dst_img = img_dir / filename
            shutil.copy2(src, dst_img)
            
            # Convertir anotaciones
            yolo_lines = []
            for ann in anns_by_img.get(img_id, []):
                if ann.get('iscrowd', 0) or 'bbox' not in ann:
                    continue
                
                cat_id = ann['category_id']
                if cat_id not in cat_map:
                    continue
                
                bbox = self.bbox_coco_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
                if bbox:
                    cls = cat_map[cat_id]
                    yolo_lines.append(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                else:
                    self.stats['invalid_boxes'] += 1
            
            # Escribir labels
            lbl_file = lbl_dir / (Path(filename).stem + '.txt')
            with open(lbl_file, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))
            
            self.stats['processed'] += 1
        
        print(f"Procesadas: {self.stats['processed']} imágenes")
        if self.stats['missing'] > 0:
            print(f"No encontradas: {self.stats['missing']}")
        if self.stats['invalid_boxes'] > 0:
            print(f"Boxes inválidos: {self.stats['invalid_boxes']}")
        
        return class_names
    
    def create_yaml(self, class_names: List[str], splits: List[str]):
        """Crea el archivo data.yaml."""
        yaml_content = f"""# Dataset YOLO - {self.dataset_name}
# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {self.output_dir.absolute()}
train: images/train
val: images/val
"""
        if 'test' in splits:
            yaml_content += "test: images/test\n"
        
        yaml_content += "\nnames:\n"
        for idx, name in enumerate(class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nCreado: {yaml_path}")
    
    def convert(self):
        """Ejecuta la conversión completa."""
        print(f"\n{'='*60}")
        print(f"Conversión COCO → YOLO")
        print(f"{'='*60}")
        print(f"Entrada:  {self.input_dir}")
        print(f"Salida:   {self.output_dir}")
        
        # Detectar splits
        splits = self.find_coco_files()
        print(f"Splits detectados: {list(splits.keys())}")
        
        # Confirmar si existe el directorio de salida
        if self.output_dir.exists():
            files = list(self.output_dir.rglob('*.jpg')) + list(self.output_dir.rglob('*.txt'))
            if files:
                print(f"\nADVERTENCIA: El directorio de salida contiene {len(files)} archivos")
                resp = input("¿Sobrescribir? [y/N]: ").strip().lower()
                if resp != 'y':
                    print("Cancelado.")
                    return False
        
        # Convertir cada split
        class_names = None
        for split, coco_file in splits.items():
            self.stats = {'processed': 0, 'missing': 0, 'invalid_boxes': 0}
            names = self.convert_split(split, coco_file)
            if class_names is None:
                class_names = names
        
        # Crear data.yaml
        self.create_yaml(class_names, list(splits.keys()))
        
        print(f"\n{'='*60}")
        print(f"Conversión completada exitosamente")
        print(f"{'='*60}")
        print(f"\nPara entrenar:")
        print(f"  yolo train data={self.output_dir / 'data.yaml'} model=yolov8n.pt epochs=100")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convierte dataset COCO a formato YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python coco_to_yolo.py -i data/animals_coco -o data/animals_yolo -n animals
  python coco_to_yolo.py -i data/pothole_coco -o data/pothole_yolo -n pothole

El script detecta automáticamente los splits (train/val/test) y las ubicaciones
de los archivos de anotaciones COCO. Funciona con cualquier dataset en formato COCO.
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Directorio del dataset COCO')
    parser.add_argument('--output', '-o', required=True, help='Directorio de salida YOLO')
    parser.add_argument('--name', '-n', required=True, help='Nombre del dataset')
    
    args = parser.parse_args()
    
    try:
        converter = CocoToYoloConverter(args.input, args.output, args.name)
        success = converter.convert()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nCancelado por el usuario")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())