import logging
import argparse
from pathlib import Path
from incremental_pipeline import (
    sha1_of_file,
    load_hash_index,
    save_hash_index,
    ensure_dirs,
    find_image_files
)
import shutil
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("staged_incremental")


class ModelVersionManager:
    """Model Version Manager"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.version_file = self.models_dir / "version_history.json"
        self.history = self._load_history()

    def _load_history(self) -> dict:
        """Load version history"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {
            "current_version": 0,
            "versions": []
        }

    def _save_history(self):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_next_version(self) -> int:
        """Get the next version number"""
        return self.history["current_version"] + 1

    def get_current_version(self) -> int:
        """Get the current version number"""
        return self.history["current_version"]

    def get_version_dir(self, version: int) -> Path:
        """Get the specified version directory"""
        return self.models_dir / f"v{version}"

    def get_current_model_path(self) -> Path:
        """Obtain the current best model path"""
        current = self.history["current_version"]
        if current == 0:
            return None
        return self.get_version_dir(current) / "weights" / "best.pt"

    def create_version(self, metrics: dict, dataset_info: dict, temp_train_dir: Path) -> int:
        """Create a new version and move the training results"""
        version = self.get_next_version()
        version_dir = self.get_version_dir(version)

        # Move the training results to the version directory
        if temp_train_dir.exists():
            if version_dir.exists():
                shutil.rmtree(version_dir)
            shutil.move(str(temp_train_dir), str(version_dir))
            logger.info(f"✅ Model saved to {version_dir}")
        else:
            logger.error(f"❌ Training directory not found: {temp_train_dir}")
            return None

        # Record version information
        version_info = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "dataset": dataset_info,
            "model_path": str(version_dir / "weights" / "best.pt")
        }

        self.history["versions"].append(version_info)
        self.history["current_version"] = version
        self._save_history()

        logger.info(f"📦 Created version v{version}")
        return version

    def compare_versions(self, v1: int, v2: int):
        """Compare the two versions and print them"""
        versions = {v["version"]: v for v in self.history["versions"]}

        if v1 not in versions or v2 not in versions:
            return

        logger.info(f"\n📈 Comparing v{v1} vs v{v2}:")

        metrics1 = versions[v1]["metrics"]
        metrics2 = versions[v2]["metrics"]

        for key in metrics1:
            if key in metrics2 and isinstance(metrics1[key], (int, float)):
                val1 = metrics1[key]
                val2 = metrics2[key]
                diff = val2 - val1
                percent = (diff / val1 * 100) if val1 != 0 else 0

                symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                logger.info(f"   {symbol} {key}: {val1:.4f} → {val2:.4f} ({percent:+.2f}%)")

    def update_symlinks(self, version: int):
        """Update symbolic links"""
        try:
            # Create/update the latest link
            latest_link = self.models_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(f"v{version}", target_is_directory=True)

            # Create/update the best_latest.pt link
            best_latest = self.models_dir / "best_latest.pt"
            if best_latest.exists() or best_latest.is_symlink():
                best_latest.unlink()
            best_latest.symlink_to(f"v{version}/weights/best.pt")

            logger.info(f"🔗 Updated symlinks to v{version}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to create symlinks: {e}")
            logger.info(f"   You can manually use: {self.get_version_dir(version)}/weights/best.pt")

    def print_version_history(self):
        """Print version history"""
        logger.info("\n" + "=" * 70)
        logger.info("📚 MODEL VERSION HISTORY")
        logger.info("=" * 70)

        if not self.history["versions"]:
            logger.info("No versions yet.")
            logger.info("=" * 70)
            return

        current = self.history["current_version"]
        logger.info(f"\n🎯 Current Version: v{current}\n")

        for v_info in self.history["versions"]:
            version = v_info["version"]
            timestamp = v_info["timestamp"]
            metrics = v_info["metrics"]

            marker = "⭐" if version == current else "  "
            logger.info(f"{marker} Version {version}")
            logger.info(f"   📅 {timestamp}")

            if metrics:
                logger.info(f"   📊 Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"      - {key}: {value:.4f}")
                    else:
                        logger.info(f"      - {key}: {value}")
            logger.info("")

        logger.info("=" * 70)


def stage1_prepare_and_label(
        new_data: Path,
        base_model: Path,
        output_dir: Path,
        labels_json: str = None
):
    from utils.yolo_utils import predict_images_to_cvat_xml

    output_dir = Path(output_dir)
    dataset_root = output_dir / "dataset"
    review_dir = output_dir / "review"
    hash_index_file = dataset_root / "hash_index.json"

    # Create an audit directory
    review_images = review_dir / "images"
    review_xml_dir = review_dir / "xmls"
    ensure_dirs([review_images, review_xml_dir, dataset_root])

    logger.info("=" * 70)
    logger.info("🚀 STAGE 1: DATA PREPARATION & AUTO-LABELING")
    logger.info("=" * 70)

    # Load the de-duplicated index
    hash_idx = load_hash_index(hash_index_file)
    existing_hashes = set(hash_idx.keys())

    # Discover new images
    all_new_images = find_image_files(new_data)
    if not all_new_images:
        logger.info("❌ No new images found. Exiting.")
        return False

    logger.info(f"📁 Found {len(all_new_images)} new images")

    # duplicate removal
    new_image_files = []
    for p in all_new_images:
        h = sha1_of_file(p)
        if h in existing_hashes:
            logger.info(f"⏭️  Skipping duplicate: {p.name}")
            continue
        dst = review_images / p.name
        shutil.copy2(p, dst)
        new_image_files.append(dst)
        hash_idx[h] = f"review/images/{p.name}"

    if not new_image_files:
        logger.info("❌ No new images after deduplication. Exiting.")
        return False

    logger.info(f"✅ {len(new_image_files)} images prepared for labeling")
    save_hash_index(hash_index_file, hash_idx)

    # Automatic annotation
    logger.info("\n" + "-" * 70)
    logger.info("🏷️  Generating automatic labels...")
    logger.info("-" * 70)

    xml_output = review_xml_dir / "staged.xml"
    predict_images_to_cvat_xml(
        str(base_model),
        str(review_images),
        str(xml_output),
        labels_json
    )

    # Save the status file
    state_file = review_dir / "stage1_complete.json"
    state = {
        "stage": 1,
        "completed": True,
        "new_images_count": len(new_image_files),
        "xml_path": str(xml_output),
        "images_path": str(review_images),
        "next_step": "Review and correct labels in CVAT, then run stage2"
    }
    with state_file.open("w") as f:
        json.dump(state, f, indent=2)

    # Completion prompt
    logger.info("\n" + "=" * 70)
    logger.info("✅ STAGE 1 COMPLETED - WAITING FOR MANUAL REVIEW")
    logger.info("=" * 70)
    logger.info("")
    logger.info("📋 Next Steps:")
    logger.info("")
    logger.info("1. 📤 Upload to CVAT for review:")
    logger.info(f"   Images: {review_images}")
    logger.info(f"   XML:    {xml_output}")
    logger.info("")
    logger.info("2. ✏️  Review and correct labels in CVAT")
    logger.info("")
    logger.info("3. 📥 Export corrected XML from CVAT")
    logger.info(f"   Save as: {xml_output}")
    logger.info(f"   (Replace the original file)")
    logger.info("")
    logger.info("4. ▶️  Run Stage 2:")
    logger.info(f"   python train_incremental.py --stage 2 --output_dir {output_dir}")
    logger.info("")
    logger.info("=" * 70)

    return True


def fix_cvat_xml_labels(xml_path: Path, labels_json_path: Path):
    import json
    import xml.etree.ElementTree as ET

    # 1. Read the correct tag mapping
    with open(labels_json_path, 'r') as f:
        correct_labels = json.load(f)

    sorted_items = sorted(correct_labels.items(), key=lambda x: int(x[0]))
    label_list = [name for id_str, name in sorted_items]

    logger.info(f"🔧 Fixing XML labels order to match {labels_json_path}")
    logger.info(f"   Total classes: {len(label_list)}")
    logger.info(f"   First 5: {label_list[:5]}")

    # 2. Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 3. Find the <labels> element
    meta = root.find(".//meta")
    if meta is None:
        logger.error("❌ No <meta> found in XML")
        return False

    task = meta.find("task")
    if task is None:
        # If there is no task, directly look for labels under meta
        labels_elem = meta.find("labels")
    else:
        labels_elem = task.find("labels")

    if labels_elem is None:
        logger.error("❌ No <labels> found in XML")
        return False

    # 4. Delete the original tag definitions
    labels_elem.clear()

    # 5. Re-add the labels in the correct order
    for i, label_name in enumerate(label_list):
        label_tag = ET.SubElement(labels_elem, "label")
        name_tag = ET.SubElement(label_tag, "name")
        name_tag.text = label_name

        # Add other necessary child elements (required by CVAT)
        ET.SubElement(label_tag, "color").text = "#000000"
        ET.SubElement(label_tag, "type").text = "rectangle"
        attributes_tag = ET.SubElement(label_tag, "attributes")

    # 6. Save the repaired XML
    backup_path = xml_path.with_suffix('.xml.backup')
    import shutil
    shutil.copy2(xml_path, backup_path)

    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    logger.info(f"✅ XML labels order fixed")
    logger.info(f"   Backup saved to: {backup_path}")

    return True


def stage2_augment_and_train(
        output_dir: Path,
        base_model: Path,
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
    from utils.augment_utils import xml_to_yolo_labels_and_augment
    import random

    output_dir = Path(output_dir)
    dataset_root = output_dir / "dataset"
    review_dir = output_dir / "review"
    hash_index_file = dataset_root / "hash_index.json"

    # Initialize the version manager
    version_manager = ModelVersionManager(output_dir)

    logger.info("=" * 70)
    logger.info("🚀 STAGE 2: AUGMENTATION & TRAINING (WITH VERSION MANAGEMENT)")
    logger.info("=" * 70)

    # Display version history
    version_manager.print_version_history()

    # Check whether Stage 1 is completed
    state_file = review_dir / "stage1_complete.json"
    if not state_file.exists():
        logger.error("❌ Stage 1 not completed! Please run stage 1 first.")
        return False

    with state_file.open("r") as f:
        state = json.load(f)

    review_images = Path(state["images_path"])
    xml_path = Path(state["xml_path"])

    if not xml_path.exists():
        logger.error(f"❌ XML file not found: {xml_path}")
        logger.error("   Please ensure the corrected XML is saved at this location.")
        return False

    logger.info("\n" + "=" * 70)
    logger.info("🏷️  INTELLIGENT LABEL MANAGEMENT SYSTEM (PRE-PROCESSING)")
    logger.info("=" * 70)

    # 1. Read the existing labels.json (if any)
    existing_labels = {}
    labels_json_path = dataset_root / "labels.json"

    if labels_json and Path(labels_json).exists():
        # The user provided labels_json
        labels_json_path = Path(labels_json)
        logger.info(f"\n📂 Loading existing labels from: {labels_json_path}")
        try:
            with open(labels_json_path, 'r', encoding='utf-8') as f:
                existing_labels = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"   ✅ Found {len(existing_labels)} existing classes")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load {labels_json_path}: {e}")
    elif labels_json_path.exists():
        # There is already labels.json in the dataset directory
        logger.info(f"\n📂 Loading existing labels from: {labels_json_path}")
        try:
            with open(labels_json_path, 'r', encoding='utf-8') as f:
                existing_labels = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"   ✅ Found {len(existing_labels)} existing classes")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load {labels_json_path}: {e}")
    else:
        logger.info("\n📂 No existing labels.json found")

    # 2. Extract labels from XML
    logger.info(f"\n📄 Extracting labels from XML: {xml_path}")
    xml_labels = []
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract from the <labels> section
        for label_elem in root.findall(".//labels/label"):
            name_elem = label_elem.find("name")
            if name_elem is not None and name_elem.text:
                label_name = name_elem.text.strip()
                if label_name and label_name not in xml_labels:
                    xml_labels.append(label_name)

        logger.info(f"   ✅ Extracted {len(xml_labels)} classes from XML")

    except Exception as e:
        logger.error(f"   ❌ Failed to parse XML: {e}")
        return False

    if not xml_labels:
        logger.error("   ❌ No labels found in XML!")
        return False

    # 3. Intelligent merge
    logger.info(f"\n🔄 Merging labels...")
    existing_name_to_id = {v: k for k, v in existing_labels.items()}
    merged_labels = {}
    new_labels = []

    for xml_label in xml_labels:
        if xml_label in existing_name_to_id:
            original_id = existing_name_to_id[xml_label]
            merged_labels[original_id] = xml_label
            logger.info(f"   ✓ Kept: {original_id} -> {xml_label}")
        else:
            new_labels.append(xml_label)

    if merged_labels:
        next_id = max(merged_labels.keys()) + 1
    else:
        next_id = 0

    for new_label in new_labels:
        merged_labels[next_id] = new_label
        logger.info(f"   ✨ Added: {next_id} -> {new_label}")
        next_id += 1

    # 4. Save labels.json
    logger.info(f"\n💾 Saving labels to: {labels_json_path}")
    try:
        labels_to_save = {str(k): v for k, v in sorted(merged_labels.items())}
        dataset_root.mkdir(parents=True, exist_ok=True)
        with open(labels_json_path, 'w', encoding='utf-8') as f:
            json.dump(labels_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ Saved {len(merged_labels)} classes")
        labels_json = str(labels_json_path)
    except Exception as e:
        logger.error(f"   ❌ Failed to save labels.json: {e}")
        return False

    logger.info("=" * 70)

    if labels_json and Path(labels_json).exists():
        logger.info("\n" + "=" * 70)
        logger.info("🔧 FIXING CVAT XML LABELS ORDER")
        logger.info("=" * 70)
        fix_result = fix_cvat_xml_labels(xml_path, Path(labels_json))
        if not fix_result:
            logger.error("❌ Failed to fix XML labels order")
            return False

    logger.info(f"✅ Using corrected labels from: {xml_path}")
    logger.info(f"   Images: {review_images}")
    logger.info(f"   Count: {state['new_images_count']}")

    # Prepare the Table of Contents
    original_dir = dataset_root / "original"
    augmented_dir = dataset_root / "augmented"
    images_train_dir = dataset_root / "images/train"
    images_val_dir = dataset_root / "images/val"
    labels_train_dir = dataset_root / "labels/train"
    labels_val_dir = dataset_root / "labels/val"
    vis_dir = dataset_root / "vis"
    tmp_aug_dir = dataset_root / "tmp_augment"

    ensure_dirs([
        original_dir / "images", original_dir / "labels",
        augmented_dir / "images", augmented_dir / "labels",
        images_train_dir, images_val_dir,
        labels_train_dir, labels_val_dir,
        vis_dir
    ])

    # Step1: data augmentation
    logger.info("\n" + "-" * 70)
    logger.info("🎨 Augmenting data...")
    logger.info("-" * 70)

    if tmp_aug_dir.exists():
        shutil.rmtree(tmp_aug_dir)
    tmp_aug_dir.mkdir(parents=True, exist_ok=True)

    xml_to_yolo_labels_and_augment(
        xml_files=[str(xml_path)],
        raw_images_root=str(review_images),
        output_dir=str(tmp_aug_dir),
        aug_output_dir=str(tmp_aug_dir),
        vis_output_dir=str(vis_dir),
        num_aug_per_image=num_aug_per_image,
        labels_json=labels_json
    )

    logger.info("✅ Augmentation completed")

    # Step2: Integrate the dataset
    logger.info("\n" + "-" * 70)
    logger.info("📦 Integrating into dataset...")
    logger.info("-" * 70)

    hash_idx = load_hash_index(hash_index_file)
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

        h = sha1_of_file(tgt_img_path)
        hash_idx[h] = str(tgt_img_path.relative_to(dataset_root))

    save_hash_index(hash_index_file, hash_idx)
    shutil.rmtree(tmp_aug_dir)

    logger.info("✅ Dataset integration completed")

    # Step3: Build train/val split
    logger.info("\n" + "-" * 70)
    logger.info("📊 Building train/val split...")
    logger.info("-" * 70)

    all_images = (
            list((original_dir / "images").glob("*.*")) +
            list((augmented_dir / "images").glob("*.*"))
    )
    random.shuffle(all_images)
    split_idx = int(len(all_images) * split_ratio)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    # Clear out the old divisions
    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()

    def copy_image_and_label(src_img: Path, dest_img_dir: Path, dest_lbl_dir: Path):
        orig_lbl = original_dir / "labels" / f"{src_img.stem}.txt"
        aug_lbl = augmented_dir / "labels" / f"{src_img.stem}.txt"
        possible_label = orig_lbl if orig_lbl.exists() else (aug_lbl if aug_lbl.exists() else None)
        shutil.copy2(src_img, dest_img_dir / src_img.name)
        if possible_label:
            shutil.copy2(possible_label, dest_lbl_dir / possible_label.name)

    for img in train_list:
        copy_image_and_label(img, images_train_dir, labels_train_dir)
    for img in val_list:
        copy_image_and_label(img, images_val_dir, labels_val_dir)

    logger.info(f"✅ Train: {len(train_list)}, Val: {len(val_list)}")


    # Step4: Generate data.yaml (Intelligent Category Name Management)
    logger.info("\n" + "-" * 70)
    logger.info("📝 Generating data.yaml with intelligent class name management...")
    logger.info("-" * 70)

    # 1. Obtain all the used category ids from the tag file
    label_files = list(labels_train_dir.glob("*.txt")) + list(labels_val_dir.glob("*.txt"))
    classes = set()
    for lf in label_files:
        with lf.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    classes.add(int(float(parts[0])))

    num_classes = max(classes) + 1 if classes else 0
    logger.info(f"   📊 Found {len(classes)} classes in labels: {sorted(classes)}")

    # 2. Initialize the category name mapping
    names_map = {}

    # 3. Method 1: Load from labels.json (highest priority)
    if labels_json and Path(labels_json).exists():
        try:
            with open(labels_json, "r", encoding="utf-8") as f:
                js = json.load(f)
                names_map = {int(k): v for k, v in js.items()}
                logger.info(f"   ✅ Loaded {len(names_map)} class names from labels.json")
                for k, v in sorted(names_map.items()):
                    logger.info(f"      {k}: {v}")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load labels.json: {e}")

    # 4. Method 2: Extract the category name (secondary priority) from the CVAT XML
    if not names_map and xml_path.exists():
        logger.info("   📄 Attempting to extract class names from CVAT XML...")
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()

            xml_labels = []

            # Try to read from meta/task/labels
            for label_elem in root.findall(".//meta/task/labels/label"):
                name_elem = label_elem.find("name")
                if name_elem is not None and name_elem.text:
                    xml_labels.append(name_elem.text)

            # If not found above, try reading directly from labels
            if not xml_labels:
                for label_elem in root.findall(".//labels/label"):
                    name_elem = label_elem.find("name")
                    if name_elem is not None and name_elem.text:
                        xml_labels.append(name_elem.text)

            if xml_labels:
                names_map = {i: name for i, name in enumerate(xml_labels)}
                logger.info(f"   ✅ Extracted {len(names_map)} class names from XML:")
                for k, v in sorted(names_map.items()):
                    logger.info(f"      {k}: {v}")
            else:
                logger.warning("   ⚠️  No class names found in XML")

        except Exception as e:
            logger.warning(f"   ⚠️  Could not parse XML: {e}")

    # 5. Method 3: Inherit the category name from the old model (catch-all strategy)
    if not names_map or len(names_map) < num_classes:
        logger.info("   🔄 Inheriting class names from previous model...")
        current_model = version_manager.get_current_model_path()

        if current_model and current_model.exists():
            try:
                from ultralytics import YOLO
                old_model = YOLO(str(current_model))

                if hasattr(old_model, 'names') and old_model.names:
                    old_names = old_model.names
                    logger.info(f"   ✅ Found {len(old_names)} class names in v{version_manager.get_current_version()}")

                    for class_id in range(num_classes):
                        if class_id not in names_map:
                            if class_id < len(old_names):
                                names_map[class_id] = old_names[class_id]
                                logger.info(f"      ✓ Inherited: {class_id} -> {old_names[class_id]}")
                            else:
                                names_map[class_id] = f"class_{class_id}"
                                logger.warning(
                                    f"      ⚠️  New class: {class_id} -> class_{class_id} (Please update labels.json!)")
                else:
                    logger.warning("   ⚠️  Previous model has no class names")

            except Exception as e:
                logger.warning(f"   ⚠️  Could not load previous model: {e}")
        else:
            logger.info("   ℹ️  No previous model found (this is v1)")

    # 6. Ultimate fallback: Use the default naming
    for i in range(num_classes):
        if i not in names_map:
            names_map[i] = f"class_{i}"
            logger.warning(f"   ⚠️  Using default name for class {i}: class_{i}")

    # 7. Display the final category mapping
    logger.info(f"\n   📋 Final class mapping ({num_classes} classes):")
    for i in sorted(names_map.keys()):
        logger.info(f"      {i}: {names_map[i]}")

    # 8. Save the category mapping to labels.json (for next use)
    labels_json_path = dataset_root / "labels.json"
    try:
        with open(labels_json_path, "w", encoding="utf-8") as f:
            json.dump(names_map, f, indent=2, ensure_ascii=False)
        logger.info(f"\n   💾 Saved class mapping to: {labels_json_path}")
        logger.info(f"      (This will be used for next training iteration)")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not save labels.json: {e}")

    # 9. Generate data.yaml
    data_yaml = dataset_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.resolve()}\n")
        f.write("train: images/train\nval: images/val\nnames:\n")
        for i in range(num_classes):
            f.write(f"  {i}: {names_map.get(i, f'class_{i}')}\n")

    logger.info(f"   ✅ data.yaml created with {num_classes} classes")

    # Step5: Training (Using Version Management)
    logger.info("\n" + "-" * 70)
    logger.info("🤖 Starting training...")
    logger.info("-" * 70)

    from ultralytics import YOLO

    # Get the next version number
    next_version = version_manager.get_next_version()
    logger.info(f"📦 Training version: v{next_version}")

    # Determine the initial model
    current_model = version_manager.get_current_model_path()
    if current_model and current_model.exists():
        init_model = current_model
        logger.info(f"📂 Loading from v{version_manager.get_current_version()}: {init_model}")
    else:
        init_model = base_model
        logger.info(f"📂 Loading base model: {init_model}")

    model = YOLO(str(init_model))

    # Train to the temporary directory
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    temp_train_name = f"temp_v{next_version}"

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(models_dir),
        name=temp_train_name,
        exist_ok=True
    )

    temp_train_dir = models_dir / temp_train_name
    produced_model = temp_train_dir / "weights" / "best.pt"

    if not produced_model.exists():
        logger.error(f"❌ Training failed: {produced_model} not found")
        return False

    logger.info(f"✅ Training completed: {produced_model}")

    # Step6: Test set verification
    metrics = {}

    if enable_test_validation and test_set_dir:
        test_set_dir = Path(test_set_dir)
        test_yaml = test_set_dir / "data.yaml"

        if test_yaml.exists():
            logger.info("\n" + "=" * 70)
            logger.info("🧪 FIXED TEST SET VALIDATION")
            logger.info("=" * 70)

            test_model = YOLO(str(produced_model))
            test_results = test_model.val(data=str(test_yaml))

            metrics = {
                "mAP50": float(test_results.box.map50),
                "mAP50-95": float(test_results.box.map),
                "precision": float(test_results.box.mp),
                "recall": float(test_results.box.mr)
            }

            logger.info(f"\n📊 Test Results:")
            logger.info(f"   mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"   mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall: {metrics['recall']:.4f}")

    # Step7: Create a version and save it
    dataset_info = {
        "new_images": state['new_images_count'],
        "total_train": len(train_list),
        "total_val": len(val_list),
        "epochs": epochs
    }

    new_version = version_manager.create_version(metrics, dataset_info, temp_train_dir)

    if new_version is None:
        logger.error("❌ Failed to create version")
        return False

    # Update Symbolic links
    version_manager.update_symlinks(new_version)

    # Compared with the previous version
    if new_version > 1:
        version_manager.compare_versions(new_version - 1, new_version)

    # Display the latest version history
    version_manager.print_version_history()

    # Clean up the review directory
    logger.info("\n" + "-" * 70)
    logger.info("🧹 Cleaning up review directory...")
    logger.info("-" * 70)

    if review_dir.exists():
        shutil.rmtree(review_dir)
        logger.info("✅ Review directory cleaned")

    logger.info("\n" + "=" * 70)
    logger.info(f"✅ STAGE 2 COMPLETED - VERSION v{new_version} CREATED")
    logger.info("=" * 70)
    logger.info(f"\n📂 Model location:")
    logger.info(f"   - Version dir: {version_manager.get_version_dir(new_version)}")
    logger.info(f"   - Best model: {version_manager.get_version_dir(new_version)}/weights/best.pt")
    logger.info(f"   - Latest link: {models_dir}/latest -> v{new_version}")
    logger.info(f"   - Quick access: {models_dir}/best_latest.pt")
    logger.info("\n" + "=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(description="Staged Incremental Training with Manual Review & Version Management")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                        help="Stage to run: 1 (prepare & label) or 2 (augment & train)")
    parser.add_argument("--new_data", type=str, default="new_data/",
                        help="New data directory (Stage 1 only)")
    parser.add_argument("--base_model", type=str, default="base.pt",
                        help="Base model path")
    parser.add_argument("--output_dir", type=str, default="incremental_output",
                        help="Output directory")
    parser.add_argument("--test_set", type=str, default="test_set",
                        help="Test set directory")
    parser.add_argument("--labels_json", type=str, default=None,
                        help="Labels JSON file")
    parser.add_argument("--num_aug", type=int, default=10,
                        help="Number of augmentations per image")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/val split ratio")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--auto_adopt_threshold", type=float, default=0.0,
                        help="Auto adopt threshold (mAP improvement %)")
    parser.add_argument("--disable_test_validation", action="store_true",
                        help="Disable test set validation")

    args = parser.parse_args()

    if args.stage == 1:
        success = stage1_prepare_and_label(
            new_data=Path(args.new_data),
            base_model=Path(args.base_model),
            output_dir=Path(args.output_dir),
            labels_json=args.labels_json
        )
    else:  # stage == 2
        success = stage2_augment_and_train(
            output_dir=Path(args.output_dir),
            base_model=Path(args.base_model),
            num_aug_per_image=args.num_aug,
            split_ratio=args.split,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            labels_json=args.labels_json,
            conf=args.conf,
            test_set_dir=Path(args.test_set),
            auto_adopt_threshold=args.auto_adopt_threshold,
            enable_test_validation=not args.disable_test_validation
        )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())