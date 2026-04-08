# Dataset Conversion Scripts

Simple, robust scripts for converting datasets to YOLO format with comprehensive validation.

## Scripts Overview

### 1. coco_to_yolo.py
Convert any COCO format dataset to YOLO format.

**Features:**
- Auto-detects train/val/test splits
- Auto-detects image directory structure
- Validates all coordinates and dimensions
- Generates data.yaml automatically
- Reports conversion statistics

**Usage:**
```bash
python scripts/coco_to_yolo.py --input <coco_dataset_dir> --output <yolo_output_dir> --name <dataset_name>
```

**Examples:**
```bash
# Animals dataset
python scripts/coco_to_yolo.py --input data/animals --output data/animals_yolo --name animals

# Pothole dataset
python scripts/coco_to_yolo.py --input data/pothole_coco --output data/pothole_yolo --name pothole
```

**Expected Input Structure:**
```
coco_dataset/
├── train/
│   ├── _annotations.coco.json (or labels.json)
│   └── *.jpg (or images/*.jpg)
├── val/
│   ├── _annotations.coco.json (or labels.json)
│   └── *.jpg (or images/*.jpg)
└── test/ (optional)
    ├── _annotations.coco.json (or labels.json)
    └── *.jpg (or images/*.jpg)
```

---

### 2. validate_yolo_dataset.py
Validate YOLO format datasets for training readiness.

**Features:**
- Validates directory structure
- Validates data.yaml format
- Checks image-label file pairing
- Validates coordinate ranges [0, 1]
- Reports comprehensive statistics

**Usage:**
```bash
python scripts/validate_yolo_dataset.py --dataset <yolo_dataset_dir>
```

**Examples:**
```bash
# Validate animals dataset
python scripts/validate_yolo_dataset.py --dataset data/animals_yolo

# Validate pothole dataset
python scripts/validate_yolo_dataset.py --dataset data/pothole_yolo
```

**Expected Input Structure:**
```
yolo_dataset/
├── data.yaml
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── labels/
    ├── train/*.txt
    └── val/*.txt
```

---

### 3. convert_coco_to_yolo.py (Legacy)
Original conversion script - kept for reference.
**Deprecated:** Use `coco_to_yolo.py` instead.

---

### 4. download_roboflow_animals.py
Download animals dataset from Roboflow.

**Usage:**
```bash
python scripts/download_roboflow_animals.py
```

---

## Validation Rules

### COCO Format Requirements
- Valid JSON structure
- Required fields: `images`, `annotations`, `categories`
- Each image must have: `id`, `file_name`, `width`, `height`
- Each annotation must have: `id`, `image_id`, `category_id`, `bbox`
- Bbox format: `[x_min, y_min, width, height]` in pixels

### YOLO Format Requirements
- One `.txt` file per image
- Each line: `class_id x_center y_center width height`
- All coordinates normalized to [0, 1]
- Coordinate validation:
  - 0 ≤ x_center ≤ 1
  - 0 ≤ y_center ≤ 1
  - 0 < width ≤ 1
  - 0 < height ≤ 1

### data.yaml Requirements
- Must specify: `path`, `train`, `val`
- Optional: `test`
- Must define class names with indices

---

## Error Handling

### Common Issues and Solutions

#### Issue: "Image not found"
**Cause:** Images in subdirectory not detected
**Solution:** Script auto-checks `images/`, `img/`, `imgs/`, `data/` subdirectories

#### Issue: "Invalid bbox - out of bounds"
**Cause:** Coordinate outside [0, 1] range or negative dimensions
**Solution:** Bbox automatically skipped, warning displayed

#### Issue: "Missing COCO annotation file"
**Cause:** File name doesn't match expected patterns
**Solution:** Use `_annotations.coco.json`, `labels.json`, or `instances_*.json`

---

## Output Format

### YOLO data.yaml Example
```yaml
# YOLO Dataset Configuration
# Generated from COCO format: animals

path: D:\road_obstacle_detection\data\animals_yolo
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: dog-camel-deer
  1: Animals
```

### YOLO Label File Example (.txt)
```
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
```
Format: `class_id x_center y_center width height`

---

## Conversion Statistics

### Animals Dataset (✅ Completed)
- Source: COCO format, 970 images
- Output: YOLO format, 1,940 images (with splits)
- Classes: 2 (dog-camel-deer, Animals)
- Splits: train/val/test
- Status: ✅ Validated, ready for training

### Pothole Dataset (✅ Completed)
- Source: COCO format, 665 images
- Output: YOLO format, 1,330 images (with splits)
- Classes: 1 (pothole)
- Splits: train/val
- Status: ✅ Validated, ready for training

### BDD100K Dataset (⏳ Pending)
- Source: Proprietary format, ~10k images
- Output: TBD
- Classes: Multiple (car, person, traffic sign, etc.)
- Status: ⏳ Awaiting Phase 2 conversion

---

## Training

After conversion and validation, train with:

```bash
# YOLOv8 Nano (fastest)
yolo train data=data/animals_yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640

# YOLOv8 Small (balanced)
yolo train data=data/animals_yolo/data.yaml model=yolov8s.pt epochs=100 imgsz=640

# YOLOv8 Medium (more accurate)
yolo train data=data/animals_yolo/data.yaml model=yolov8m.pt epochs=100 imgsz=640
```

---

## Design Philosophy

### Simplicity
- No complex frameworks
- Clear, straightforward logic
- Easy to understand and modify

### Validation
- Multiple validation layers
- Clear error messages
- Data integrity ensured

### Maintainability
- Well-documented code
- Single responsibility functions
- Easy to debug and extend

### No Over-Engineering
- Just enough validation
- No unnecessary abstractions
- Avoid "hard-to-detect failures"

---

## Requirements

```bash
pip install pycocotools
```

Standard library only (json, pathlib, shutil, argparse).

---

## Contributing

When adding new features:
1. Keep it simple
2. Validate data integrity
3. Provide clear error messages
4. Update documentation
5. Test with multiple datasets

---

## Support

For issues or questions, refer to:
- `CONVERSION_SUMMARY.md` - Detailed conversion results
- `DATASET_COMPATIBILITY_ANALYSIS.md` - Dataset analysis and strategy
- Individual script help: `python scripts/<script>.py --help`
