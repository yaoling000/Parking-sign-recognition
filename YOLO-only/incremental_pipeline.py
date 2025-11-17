"""
Incremental dataset + model update pipeline with fixed test set validation.

- new_data: folder with new images
- base_model: initial YOLO model (used if no best_latest.pt)
- dataset_root: accumulated dataset (images/labels/train/val + data.yaml)
- models_dir: where trained models are saved
- test_set: fixed test set for model comparison (optional)
"""

import argparse, logging, shutil, random, json, hashlib
from pathlib import Path
from typing import List, Set, Dict

try:
    from utils.yolo_utils import predict_images_to_cvat_xml
    from utils.augment_utils import xml_to_yolo_labels_and_augment
except Exception:
    raise ImportError("Ensure utils/yolo_utils.py and utils/augment_utils.py exist")

logger = logging.getLogger("incremental_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# --- helpers ---
def sha1_of_file(p: Path, block_size: int = 65536) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def load_hash_index(path: Path) -> Dict[str, str]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_hash_index(path: Path, idx: Dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx, indent=2), encoding="utf-8")


def ensure_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def find_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


# --------------------- Fixed test set verification ---------------------
def evaluate_on_test_set(
        model_path: Path,
        test_images_dir: Path,
        test_labels_dir: Path = None,
        labels_json: str = None,
        conf: float = 0.25,
        imgsz: int = 640
) -> dict:

    from ultralytics import YOLO

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return {}

    if not test_images_dir.exists():
        logger.warning(f"Test images directory not found: {test_images_dir}")
        return {}

    logger.info(f"Evaluating model on test set: {model_path.name}")
    model = YOLO(str(model_path))

    # If there is a label, use the official val() method
    if test_labels_dir and test_labels_dir.exists():
        # Create a temporary data.yaml
        temp_yaml = test_images_dir.parent / "test_data_temp.yaml"

        # Collect category information
        label_files = list(test_labels_dir.glob("*.txt"))
        classes = set()
        for lf in label_files:
            try:
                with lf.open("r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            classes.add(int(float(parts[0])))
            except Exception as e:
                logger.warning(f"Error reading label {lf}: {e}")

        num_classes = max(classes) + 1 if classes else 1
        names_map = {i: f"class_{i}" for i in range(num_classes)}

        # If there is labels.json, use the real category name
        if labels_json:
            try:
                with open(labels_json, "r", encoding="utf-8") as f:
                    names_map = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                pass

        # Write to the temporary yaml
        with temp_yaml.open("w", encoding="utf-8") as f:
            f.write(f"path: {test_images_dir.parent.resolve()}\n")
            f.write(f"test: {test_images_dir.name}\n")
            f.write("names:\n")
            for i in range(num_classes):
                f.write(f"  {i}: {names_map.get(i, f'class_{i}')}\n")

        try:
            # Official verification
            results = model.val(
                data=str(temp_yaml),
                split='test',
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )

            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'has_labels': True
            }

            # Clear temporary documents
            if temp_yaml.exists():
                temp_yaml.unlink()

            return metrics
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            if temp_yaml.exists():
                temp_yaml.unlink()

    # If there are no labels, only inference statistics are conducted
    test_images = list(test_images_dir.glob("*.*"))
    if not test_images:
        logger.warning("No test images found")
        return {}

    results = model.predict(
        source=[str(p) for p in test_images],
        conf=conf,
        save=False,
        imgsz=imgsz,
        verbose=False
    )

    # Statistics
    total_detections = 0
    class_counts = {}
    total_confidence = 0

    for r in results:
        boxes = r.boxes
        total_detections += len(boxes)
        for box in boxes:
            cls_id = int(box.cls.item())
            conf_val = float(box.conf.item())
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_confidence += conf_val

    avg_confidence = total_confidence / total_detections if total_detections > 0 else 0

    metrics = {
        'total_detections': total_detections,
        'class_counts': class_counts,
        'avg_confidence': avg_confidence,
        'has_labels': False
    }

    return metrics


def compare_models_on_test_set(
        old_model_path: Path,
        new_model_path: Path,
        test_images_dir: Path,
        test_labels_dir: Path = None,
        labels_json: str = None,
        conf: float = 0.25,
        imgsz: int = 640,
        auto_adopt_threshold: float = 0.0
) -> bool:

    logger.info("\n" + "=" * 70)
    logger.info("📊 MODEL COMPARISON ON FIXED TEST SET")
    logger.info("=" * 70)

    # Evaluate the old model
    logger.info("\n🔵 Evaluating OLD model...")
    old_metrics = evaluate_on_test_set(
        old_model_path, test_images_dir, test_labels_dir,
        labels_json, conf, imgsz
    )

    # Evaluate the new model
    logger.info("🟢 Evaluating NEW model...")
    new_metrics = evaluate_on_test_set(
        new_model_path, test_images_dir, test_labels_dir,
        labels_json, conf, imgsz
    )

    if not old_metrics or not new_metrics:
        logger.warning("⚠️  Evaluation failed, skipping comparison")
        return False

    # Display the comparison results
    logger.info("\n" + "=" * 70)
    logger.info("📈 COMPARISON RESULTS")
    logger.info("=" * 70)

    adopt_new = False

    if old_metrics.get('has_labels') and new_metrics.get('has_labels'):
        # Comparison of complete indicators
        logger.info(f"\n{'Metric':<20} {'Old Model':<15} {'New Model':<15} {'Change':<20}")
        logger.info("-" * 70)

        metrics_to_compare = ['mAP50', 'mAP50-95', 'precision', 'recall']
        improvements = []

        for metric in metrics_to_compare:
            old_val = old_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            change = new_val - old_val
            change_pct = (change / old_val * 100) if old_val > 0 else 0

            improvements.append(change_pct)

            if change > 0:
                symbol = "✅"
            elif change < 0:
                symbol = "❌"
            else:
                symbol = "➖"

            logger.info(
                f"{metric:<20} {old_val:<15.4f} {new_val:<15.4f} "
                f"{symbol} {change:+.4f} ({change_pct:+.2f}%)"
            )

        map50_improvement = improvements[0]

        logger.info("\n" + "-" * 70)
        if map50_improvement > auto_adopt_threshold:
            logger.info(f"✅ NEW MODEL IS BETTER! (mAP50 improved by {map50_improvement:.2f}%)")
            adopt_new = True
        elif map50_improvement < -auto_adopt_threshold:
            logger.info(f"❌ OLD MODEL IS BETTER! (mAP50 decreased by {abs(map50_improvement):.2f}%)")
            adopt_new = False
        else:
            logger.info(f"➖ Models perform similarly (change: {map50_improvement:.2f}%)")
            adopt_new = map50_improvement >= 0

    else:
        # Untagged comparison
        logger.info(f"\n{'Metric':<25} {'Old Model':<15} {'New Model':<15} {'Change':<15}")
        logger.info("-" * 70)

        old_det = old_metrics.get('total_detections', 0)
        new_det = new_metrics.get('total_detections', 0)
        old_conf = old_metrics.get('avg_confidence', 0)
        new_conf = new_metrics.get('avg_confidence', 0)

        logger.info(f"{'Total Detections':<25} {old_det:<15} {new_det:<15} {new_det - old_det:+15}")
        logger.info(f"{'Avg Confidence':<25} {old_conf:<15.3f} {new_conf:<15.3f} {new_conf - old_conf:+15.3f}")

        logger.info("\n⚠️  NOTE: Without ground truth labels, adopting new model by default.")
        logger.info("   Consider adding labels to test_set/labels/ for accurate evaluation.")
        adopt_new = True

    logger.info("=" * 70 + "\n")

    return adopt_new


# --- incremental pipeline ---
def incremental_pipeline(
        new_data: Path,
        base_model: Path,
        output_dir: Path,
        num_aug_per_image: int = 10,
        split_ratio: float = 0.8,
        epochs: int = 20,
        batch: int = 8,
        imgsz: int = 640,
        labels_json: str = None,
        conf: float = 0.25,
        test_set_dir: Path = None,
        auto_adopt_threshold: float = 0.0,
        enable_test_validation: bool = True
):

    output_dir = Path(output_dir)
    dataset_root = output_dir / "dataset"
    models_dir = output_dir / "models"
    hash_index_file = dataset_root / "hash_index.json"

    # dataset dirs
    original_dir = dataset_root / "original"
    augmented_dir = dataset_root / "augmented"
    images_train_dir = dataset_root / "images/train"
    images_val_dir = dataset_root / "images/val"
    labels_train_dir = dataset_root / "labels/train"
    labels_val_dir = dataset_root / "labels/val"
    vis_dir = dataset_root / "vis"
    tmp_dir = dataset_root / "tmp_new"
    ensure_dirs([original_dir / "images", original_dir / "labels",
                 augmented_dir / "images", augmented_dir / "labels",
                 images_train_dir, images_val_dir,
                 labels_train_dir, labels_val_dir,
                 vis_dir, tmp_dir])

    # load hash index
    hash_idx = load_hash_index(hash_index_file)
    existing_hashes = set(hash_idx.keys())

    # discover new images
    all_new_images = find_image_files(new_data)
    if not all_new_images:
        logger.info("No new images found. Exiting.")
        return

    logger.info(f"Found {len(all_new_images)} new images")
    staging_images = tmp_dir / "images"
    ensure_dirs([staging_images])
    new_image_names = []
    for p in all_new_images:
        h = sha1_of_file(p)
        if h in existing_hashes:
            logger.info(f"Skipping duplicate: {p.name}")
            continue
        dst = staging_images / p.name
        shutil.copy2(p, dst)
        new_image_names.append(p.name)
        hash_idx[h] = str(dst.relative_to(dataset_root))
    if not new_image_names:
        logger.info("No new images after deduplication. Exiting.")
        save_hash_index(hash_index_file, hash_idx)
        return
    logger.info(f"{len(new_image_names)} images staged for augmentation")

    # Step1: predict XML using current best model
    tmp_xml_dir = tmp_dir / "xmls"
    ensure_dirs([tmp_xml_dir])
    tmp_xml = tmp_xml_dir / "staged.xml"
    predict_images_to_cvat_xml(str(base_model), str(staging_images), str(tmp_xml), labels_json)

    # Step2: augment
    tmp_aug_dir = tmp_dir / "aug_out"
    if tmp_aug_dir.exists():
        shutil.rmtree(tmp_aug_dir)
    tmp_aug_dir.mkdir(parents=True, exist_ok=True)
    xml_to_yolo_labels_and_augment(
        xml_files=[str(tmp_xml)],
        raw_images_root=str(staging_images),
        output_dir=str(tmp_aug_dir),
        aug_output_dir=str(tmp_aug_dir),
        vis_output_dir=str(vis_dir),
        num_aug_per_image=num_aug_per_image
    )
    logger.info("Augmentation done.")

    # Step3: move originals & augmented into dataset
    tmp_imgs = list((tmp_aug_dir / "images").glob("*.*"))
    tmp_labels = tmp_aug_dir / "labels"
    for img in tmp_imgs:
        is_aug = "_aug_" in img.name
        tgt_img_dir = augmented_dir / "images" if is_aug else original_dir / "images"
        tgt_lbl_dir = augmented_dir / "labels" if is_aug else original_dir / "labels"
        ensure_dirs([tgt_img_dir, tgt_lbl_dir])
        tgt_img_path = tgt_img_dir / img.name
        if tgt_img_path.exists():
            stem, suf = img.stem, 1
            while tgt_img_dir.joinpath(f"{stem}_{suf}{img.suffix}").exists():
                suf += 1
            tgt_img_path = tgt_img_dir / f"{stem}_{suf}{img.suffix}"
        shutil.move(str(img), str(tgt_img_path))
        lbl_src = tmp_labels / f"{img.stem}.txt"
        if lbl_src.exists():
            tgt_lbl_path = tgt_lbl_dir / lbl_src.name
            if tgt_lbl_path.exists():
                stem, suf = tgt_lbl_path.stem, 1
                while tgt_lbl_dir.joinpath(f"{stem}_{suf}.txt").exists():
                    suf += 1
                tgt_lbl_path = tgt_lbl_dir / f"{stem}_{suf}.txt"
            shutil.move(str(lbl_src), str(tgt_lbl_path))
        # update hash
        h = sha1_of_file(tgt_img_path)
        hash_idx[h] = str(tgt_img_path.relative_to(dataset_root))
    save_hash_index(hash_index_file, hash_idx)
    shutil.rmtree(tmp_dir)

    # Step4: build train/val split
    all_images = list((original_dir / "images").glob("*.*")) + list((augmented_dir / "images").glob("*.*"))
    random.shuffle(all_images)
    split_idx = int(len(all_images) * split_ratio)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    # clear old splits
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        for f in d.glob("*"):
            if f.is_file(): f.unlink()

    def copy_image_and_label(src_img: Path, dest_img_dir: Path, dest_lbl_dir: Path):
        orig_lbl = original_dir / "labels" / f"{src_img.stem}.txt"
        aug_lbl = augmented_dir / "labels" / f"{src_img.stem}.txt"
        possible_label = orig_lbl if orig_lbl.exists() else (aug_lbl if aug_lbl.exists() else None)
        shutil.copy2(src_img, dest_img_dir / src_img.name)
        if possible_label:
            shutil.copy2(possible_label, dest_lbl_dir / possible_label.name)
        else:
            logger.warning(f"No label for image {src_img}")

    for img in train_list:
        copy_image_and_label(img, images_train_dir, labels_train_dir)
    for img in val_list:
        copy_image_and_label(img, images_val_dir, labels_val_dir)
    logger.info(f"Train/Val split done: {len(train_list)} train, {len(val_list)} val")

    # Step5: generate data.yaml
    # collect class ids
    label_files = list(labels_train_dir.glob("*.txt")) + list(labels_val_dir.glob("*.txt"))
    classes = set()
    for lf in label_files:
        with lf.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts: continue
                classes.add(int(float(parts[0])))
    num_classes = max(classes) + 1 if classes else 0
    names_map = {i: f"class_{i}" for i in range(num_classes)}
    if labels_json:
        try:
            with open(labels_json, "r", encoding="utf-8") as f:
                js = json.load(f)
                names_map = {int(k): v for k, v in js.items()}
        except Exception:
            logger.warning("Failed to load labels.json, fallback to placeholder names")

    data_yaml = dataset_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.resolve()}\n")
        f.write("train: images/train\nval: images/val\nnames:\n")
        for i in range(num_classes):
            f.write(f"  {i}: {names_map.get(i, f'class_{i}')}\n")
    logger.info(f"data.yaml written at {data_yaml}")

    # Step6: continue training
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    latest_model = models_dir / "best_latest.pt"
    init_model = latest_model if latest_model.exists() else base_model

    # 保存旧模型路径用于后续对比
    old_model_for_comparison = init_model

    logger.info(f"Starting training from: {init_model}")
    model = YOLO(str(init_model))
    models_dir.mkdir(parents=True, exist_ok=True)

    res = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(models_dir),
        name="incremental",
        exist_ok=True
    )

    produced = models_dir / "incremental" / "weights" / "base.pt"
    if not produced.exists():
        logger.warning("base.pt not found after training")
        return

    # Step7: Verify and compare the model on a fixed test set (if enabled)
    adopt_new_model = True

    if enable_test_validation and test_set_dir:
        test_set_dir = Path(test_set_dir)
        test_images = test_set_dir / "images"
        test_labels = test_set_dir / "labels"

        if test_images.exists():
            logger.info("\n" + "=" * 70)
            logger.info("🧪 FIXED TEST SET VALIDATION")
            logger.info("=" * 70)

            adopt_new_model = compare_models_on_test_set(
                old_model_path=old_model_for_comparison,
                new_model_path=produced,
                test_images_dir=test_images,
                test_labels_dir=test_labels if test_labels.exists() else None,
                labels_json=labels_json,
                conf=conf,
                imgsz=imgsz,
                auto_adopt_threshold=auto_adopt_threshold
            )
        else:
            logger.warning(f"⚠️  Test set not found at {test_images}, skipping validation")
            logger.info("   To enable test validation, create test_set/images/ directory")

    # Step8: Decide whether to update the model based on the verification results
    if adopt_new_model:
        existing = sorted(models_dir.glob("best_v*.pt"))
        next_v = 1 + max(
            [int(p.stem.replace("best_v", "")) for p in existing if p.stem.replace("best_v", "").isdigit()] or [0])

        versioned_model = models_dir / f"best_v{next_v}.pt"
        latest_model_path = models_dir / "best_latest.pt"

        shutil.copy2(produced, versioned_model)
        shutil.copy2(produced, latest_model_path)

        logger.info(f"\n✅ Model updated successfully!")
        logger.info(f"   - Versioned model saved: {versioned_model.name}")
        logger.info(f"   - Latest model updated: best_latest.pt")
    else:
        logger.info(f"\n⚠️  New model NOT adopted (performance did not improve)")
        logger.info(f"   - Keeping existing best_latest.pt")
        logger.info(f"   - New model saved at: {produced}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Incremental pipeline completed successfully.")
    logger.info("=" * 70 + "\n")


# --------------------- Incremental verification---------------------
def incremental_validation(model_path: Path, val_images_dir: Path, labels_json: str = None):
    """Retain the original simple verification function"""
    from ultralytics import YOLO

    logger.info(f"Starting validation on {val_images_dir} using {model_path}")
    model = YOLO(str(model_path))
    val_images = list(Path(val_images_dir).glob("*.*"))
    if not val_images:
        logger.warning("No images found for validation.")
        return

    # Inference
    results = model.predict(
        source=[str(p) for p in val_images],
        conf=0.25,
        save=False,
        imgsz=640,
        verbose=False
    )

    # Simple statistics
    class_counts = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    # If labels.json is provided, output by name
    if labels_json:
        try:
            with open(labels_json, "r", encoding="utf-8") as f:
                names_map = {int(k): v for k, v in json.load(f).items()}
        except:
            names_map = {}
    else:
        names_map = {}

    logger.info("Validation results:")
    for cls_id, count in class_counts.items():
        name = names_map.get(cls_id, f"class_{cls_id}")
        logger.info(f"  {name} ({cls_id}): {count} detections")
    logger.info("Incremental validation finished.\n")


# --- CLI ---
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--new_data", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_aug", type=int, default=10)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--labels_json", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--test_set", default=None, help="Path to fixed test set directory")
    p.add_argument("--auto_adopt_threshold", type=float, default=0.0,
                   help="Auto-adopt new model if mAP improves by this percentage (default: 0)")
    p.add_argument("--disable_test_validation", action="store_true",
                   help="Disable test set validation")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    incremental_pipeline(
        new_data=Path(args.new_data),
        base_model=Path(args.base_model),
        output_dir=Path(args.output_dir),
        num_aug_per_image=args.num_aug,
        split_ratio=args.split,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        labels_json=args.labels_json,
        conf=args.conf,
        test_set_dir=Path(args.test_set) if args.test_set else None,
        auto_adopt_threshold=args.auto_adopt_threshold,
        enable_test_validation=not args.disable_test_validation
    )