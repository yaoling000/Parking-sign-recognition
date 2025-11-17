from pathlib import Path
import logging
from typing import Optional, List
from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from .xml_utils import save_cvat_xml

logger = logging.getLogger("yolo_utils")

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

import json

def patch_ultralytics_for_new_models():
    try:
        from ultralytics.nn import tasks

        # Save the original fuse method
        if not hasattr(tasks.DetectionModel, '_original_fuse'):
            tasks.DetectionModel._original_fuse = tasks.DetectionModel.fuse

        def safe_fuse(self, verbose=True):
            """A secure fuse method compatible with the new model"""
            if verbose:
                logger.info("Fusing layers (compatibility mode)...")

            for m in self.model.modules():
                # Check if it is Conv and has the bn attribute
                if type(m).__name__ == 'Conv' and hasattr(m, 'bn'):
                    try:
                        # Attempt fusion
                        from torch.nn.utils.fusion import fuse_conv_bn_eval
                        m.conv = fuse_conv_bn_eval(m.conv, m.bn)
                        delattr(m, 'bn')
                        m.forward = m.forward_fuse
                    except AttributeError as e:
                        # If there is no bn, skip it
                        if verbose:
                            logger.debug(f"Skipping layer without bn: {type(m)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Could not fuse layer {type(m)}: {e}")
                        continue

                # C2f module
                if type(m).__name__ == 'C2f':
                    try:
                        if hasattr(m, 'forward_fuse'):
                            m.forward = m.forward_fuse
                    except Exception:
                        continue

            self.info()
            return self

        # Replace with the secure version
        tasks.DetectionModel.fuse = safe_fuse
        logger.info("✅ Ultralytics compatibility patch applied")

    except Exception as e:
        logger.warning(f"⚠️  Could not apply compatibility patch: {e}")
        logger.warning("   Model loading may fail with newer model formats")


# Automatically apply patches when the module is imported
if HAS_ULTRALYTICS:
    patch_ultralytics_for_new_models()

def read_labels_json(path: Path) -> List[str]:
    """读取 labels.json"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "labels" in data:
        return data["labels"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data[str(i)] for i in sorted(map(int, data.keys()))]
    raise ValueError("labels.json must be dict with 'labels' key, a list, or {id: name} dict.")


def predict_images_to_cvat_xml(model_path: str,
                               img_dir: str,
                               output_xml: str,
                               labels_json: Optional[str] = None,
                               conf: float = 0.25,
                               max_workers: int = 8):

    if not HAS_ULTRALYTICS:
        raise RuntimeError("Ultralytics not installed")

    # 1. load category
    labels_list: Optional[List[str]] = None
    if labels_json:
        try:
            labels_list = read_labels_json(Path(labels_json))
            logger.info(f"✅ Loaded labels from {labels_json}: {labels_list}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load {labels_json}: {e}")

    # load model
    logger.info(f"🔧 Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # 3. if there is no labels.json, use the model 's own class
    if labels_list is None and hasattr(model, "names"):
        # model.names 是 dict {cls_id: name}
        labels_list = [str(v) for k, v in sorted(model.names.items())]
        logger.info(f"✅ Using model's built-in labels: {labels_list}")

    img_dir_p = Path(img_dir)
    images = sorted([p for p in img_dir_p.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not images:
        logger.error(f"❌ No images found in {img_dir_p}")
        return

    logger.info(f"📁 Found {len(images)} images")
    annotations = Element("annotations")

    def process_one(idx: int, img_path: Path) -> Optional[Element]:
        try:
            results = model.predict(str(img_path), conf=conf, verbose=False)
            if len(results) == 0:
                logger.warning(f"No result for {img_path}")
                return None

            img_h, img_w = results[0].orig_shape[:2]
            image_elem = Element("image", {
                "id": str(idx),
                "name": img_path.name,
                "width": str(img_w),
                "height": str(img_h)
            })

            box_count = 0
            for r in results:
                for box in getattr(r, "boxes", []):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls.item()) if hasattr(box, "cls") else 0
                    score = float(box.conf.item()) if hasattr(box, "conf") else 0.0
                    label_name = labels_list[cls_id] if labels_list and cls_id < len(labels_list) else str(cls_id)
                    ET.SubElement(image_elem, "box", {
                        "label": label_name,
                        "xtl": f"{x1:.4f}",
                        "ytl": f"{y1:.4f}",
                        "xbr": f"{x2:.4f}",
                        "ybr": f"{y2:.4f}",
                        "occluded": "0",
                        "source": "auto",
                        "score": f"{score:.4f}"
                    })
                    box_count += 1

            if box_count > 0:
                logger.info(f"   ✓ {img_path.name}: {box_count} detections")
            else:
                logger.info(f"   ○ {img_path.name}: no detections")

            return image_elem

        except Exception as e:
            logger.error(f"   ❌ Failed to process {img_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    # 4. multi-threaded batch inference
    logger.info("🔄 Processing images...")
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_one, idx, p): p for idx, p in enumerate(images)}
        for fut in as_completed(futures):
            elem = fut.result()
            if elem is not None:
                annotations.append(elem)
                processed += 1

    # 5. save XML
    save_cvat_xml(annotations, Path(output_xml))
    logger.info(f"\n✅ CVAT XML saved to: {output_xml}")
    logger.info(f"   Total images: {len(images)}")
    logger.info(f"   Successfully processed: {processed}")