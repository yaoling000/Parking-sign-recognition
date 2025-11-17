from pathlib import Path
import shutil
import logging
from typing import List, Dict, Optional
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("augment_utils")

# ---------- Tool ----------
def ensure_dir(p: Path):
    """Make sure the directory exists"""
    p.mkdir(parents=True, exist_ok=True)

def default_transform() -> A.Compose:
    """The default enhanced pipeline does not include flipping"""
    return A.Compose([
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(p=0.2),
        A.GaussNoise(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ---------- Single image processing ----------
def process_image_element(
    image_elem: ET.Element,
    raw_images_root: Path,
    output_dir: Path,
    aug_output_dir: Path,
    vis_output_dir: Path,
    global_labels: Dict[str, int],
    transform: A.Compose,
    num_aug_per_image: int = 20
):
    """Processing a single image: Generate YOLO tags + data augmentation + visualization"""
    img_name = image_elem.attrib["name"]
    img_w = int(image_elem.attrib.get("width", 0))
    img_h = int(image_elem.attrib.get("height", 0))
    src_path = raw_images_root / img_name
    if not src_path.exists():
        logger.warning(f"Image not found: {src_path}")
        return

    # Copy the original image
    dst_img = output_dir / "images" / img_name
    shutil.copy2(src_path, dst_img)

    yolo_lines = []
    bboxes = []
    labels = []

    # Handle boxes
    for box in image_elem.findall("box"):
        label = box.attrib["label"]
        cls = global_labels[label]
        xtl, ytl, xbr, ybr = map(float, (box.attrib["xtl"], box.attrib["ytl"], box.attrib["xbr"], box.attrib["ybr"]))
        x_center = (xtl + xbr) / 2 / img_w
        y_center = (ytl + ybr) / 2 / img_h
        w = (xbr - xtl) / img_w
        h = (ybr - ytl) / img_h
        yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        bboxes.append([x_center, y_center, w, h])
        labels.append(cls)

    # Handle polygons
    for poly in image_elem.findall("polygon"):
        label = poly.attrib["label"]
        cls = global_labels[label]
        points = poly.attrib["points"].split(";")
        xs = [float(p.split(",")[0]) for p in points]
        ys = [float(p.split(",")[1]) for p in points]
        xtl, xbr = min(xs), max(xs)
        ytl, ybr = min(ys), max(ys)
        x_center = (xtl + xbr) / 2 / img_w
        y_center = (ytl + ybr) / 2 / img_h
        w = (xbr - xtl) / img_w
        h = (ybr - ytl) / img_h
        yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        bboxes.append([x_center, y_center, w, h])
        labels.append(cls)

    # Write the original YOLO tag
    label_file = output_dir / "labels" / f"{Path(img_name).stem}.txt"
    with open(label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    # Data augmentation
    if len(bboxes) == 0:
        return

    img_cv = cv2.imread(str(src_path))
    for i in range(num_aug_per_image):
        aug = transform(image=img_cv, bboxes=bboxes, class_labels=labels)
        aug_img = aug["image"]
        aug_bboxes = aug["bboxes"]
        aug_labels = aug["class_labels"]

        # Save the enhanced image
        aug_img_name = f"{Path(img_name).stem}_aug_{i}.jpg"
        cv2.imwrite(str(aug_output_dir / "images" / aug_img_name), aug_img)

        # Save the enhanced tag
        aug_label_file = aug_output_dir / "labels" / f"{Path(img_name).stem}_aug_{i}.txt"
        with open(aug_label_file, "w", encoding="utf-8") as f:
            for cls_id, (x, y, w_, h_) in zip(aug_labels, aug_bboxes):
                f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w_:.6f} {h_:.6f}\n")

        # Visualization
        vis_img = aug_img.copy()
        for cls_id, (x, y, w_, h_) in zip(aug_labels, aug_bboxes):
            x1 = int((x - w_/2) * img_w)
            y1 = int((y - h_/2) * img_h)
            x2 = int((x + w_/2) * img_w)
            y2 = int((y + h_/2) * img_h)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, str(cls_id), (max(0,x1), max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        vis_name = f"{Path(img_name).stem}_aug_{i}_vis.jpg"
        cv2.imwrite(str(vis_output_dir / vis_name), vis_img)

# ---------- Batch processing function ----------
def xml_to_yolo_labels_and_augment(
        xml_files: List[str],
        raw_images_root: str,
        output_dir: str = "original_dataset",
        aug_output_dir: str = "yolo_aug_dataset",
        vis_output_dir: str = "yolo_aug_dataset/vis",
        num_aug_per_image: int = 20,
        num_workers: int = 4,
        transform: Optional[A.Compose] = None,
        labels_json: Optional[str] = None
):
    """Batch XML conversion to YOLO and enhancement"""
    import json

    if transform is None:
        transform = default_transform()

    output_dir_p = Path(output_dir)
    aug_output_dir_p = Path(aug_output_dir)
    vis_output_dir_p = Path(vis_output_dir)
    for d in [output_dir_p / "images", output_dir_p / "labels",
              aug_output_dir_p / "images", aug_output_dir_p / "labels",
              vis_output_dir_p]:
        ensure_dir(d)

    # labels.json
    if not labels_json or not Path(labels_json).exists():
        logger.error("=" * 70)
        logger.error("❌ CRITICAL ERROR: labels.json is REQUIRED!")
        logger.error("=" * 70)
        logger.error("Without labels.json, class IDs will be assigned incorrectly.")
        logger.error("This will cause complete label mismatch!")
        logger.error(f"Expected location: {labels_json}")
        raise ValueError("labels.json not found - cannot proceed safely")

    # Load the correct mapping from labels.json
    logger.info("=" * 70)
    logger.info("📋 Loading label mapping from labels.json")
    logger.info("=" * 70)

    with open(labels_json, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    # {"0": "name0", "1": "name1"} -> {"name0": 0, "name1": 1}
    global_labels = {v: int(k) for k, v in labels_data.items()}

    logger.info(f"✅ Loaded {len(global_labels)} classes from labels.json:")
    for name, id in sorted(global_labels.items(), key=lambda x: x[1])[:10]:
        logger.info(f"   {id}: {name}")
    if len(global_labels) > 10:
        logger.info(f"   ... (showing first 10 of {len(global_labels)} classes)")
    logger.info("=" * 70)

    # Parse XML files
    logger.info("\n📄 Parsing XML files...")
    parsed_trees = []
    for xml in xml_files:
        xml_p = Path(xml)
        if not xml_p.exists():
            logger.warning(f"XML not found: {xml}")
            continue
        tree = ET.parse(str(xml_p))
        parsed_trees.append((xml_p, tree))

    logger.info("\n🔍 Validating XML labels...")
    unknown_labels = set()
    all_xml_labels = set()

    for xml_p, tree in parsed_trees:
        root = tree.getroot()
        for box in root.findall(".//box"):
            label = box.attrib["label"]
            all_xml_labels.add(label)
            if label not in global_labels:
                unknown_labels.add(label)
        for poly in root.findall(".//polygon"):
            label = poly.attrib["label"]
            all_xml_labels.add(label)
            if label not in global_labels:
                unknown_labels.add(label)

    if unknown_labels:
        logger.error("=" * 70)
        logger.error("❌ VALIDATION FAILED: Unknown labels in XML!")
        logger.error("=" * 70)
        logger.error(f"Found {len(unknown_labels)} unknown labels:")
        for label in sorted(unknown_labels):
            logger.error(f"   - {label}")
        logger.error("\nAvailable labels in labels.json:")
        for label in sorted(global_labels.keys())[:20]:
            logger.error(f"   - {label}")
        if len(global_labels) > 20:
            logger.error(f"   ... (showing first 20 of {len(global_labels)} labels)")
        logger.error("=" * 70)
        raise ValueError(f"Found {len(unknown_labels)} unknown labels in XML")

    logger.info(f"✅ All {len(all_xml_labels)} XML labels are valid")
    logger.info(f"   Labels used: {sorted(all_xml_labels)[:5]}...")

    # Batch processing of images
    image_elements = []
    for xml_p, tree in parsed_trees:
        root = tree.getroot()
        for image_elem in root.findall(".//image"):
            image_elements.append((image_elem, xml_p))

    logger.info(f"\n📊 Total images to process: {len(image_elements)}")

    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(process_image_element,
                              ie,
                              Path(raw_images_root),
                              output_dir_p,
                              aug_output_dir_p,
                              vis_output_dir_p,
                              global_labels,
                              transform,
                              num_aug_per_image) for ie, _ in image_elements]
        for fut in as_completed(futures):
            try:
                _ = fut.result()
            except Exception as e:
                logger.error(f"❌ Error processing image: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise