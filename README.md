# Australian Parking Sign Recognition System

A comprehensive system for interpreting parking sign information from real street images in Australia, comparing three different approaches: YOLO-only, YOLO+OCR+NLP, and FastVLM.

## 📋 Project Overview

This project aims to build a model that can interpret parking sign information from real street images in Australia. Parking signs incorporate symbols, arrows, as well as dense text regarding duration, days, and special conditions. Our goal is not only to identify the signs, but also to explain their rules and output them in a consistent and machine-readable JSON format.

### Three Approaches Studied

We studied three methods to solve this problem:

1. **YOLO-only Pipeline** ✅ (Current Implementation)
   - Uses YOLOv8 alone to complete the entire process
   - Directly learns fine-grained categories on the signs
   - Enables rapid recognition of known categories
   - **Current Performance: mAP@0.5 > 0.90** 🎯

2. **YOLO + OCR + NLP Pipeline** (Coming Soon)
   - YOLO for sign and symbol detection
   - PaddleOCR for text extraction
   - NLP for converting text into structured output

3. **FastVLM Pipeline** (Coming Soon)
   - End-to-end vision-language model
   - Jointly processes images and text
   - RLHF optimization to reduce false outputs

### Dataset

The data used in this project was obtained from hundreds of photos taken in various locations across Sydney, with different shooting environments and perspectives. We defined a unified JSON format and evaluated the model's Precision, Recall and F1-Score on the images and fields.

## 🏗️ Repository Structure

```
Parking-sign-recognition/
├── YOLO-only/                      # ✅ YOLO-only pipeline (implemented)
│   ├── utils/                      # Utility modules
│   │   ├── augment_utils.py       # Data augmentation
│   │   ├── Version_manager.py     # Model versioning
│   │   ├── xml_utils.py           # CVAT XML processing
│   │   └── yolo_utils.py          # YOLO utilities
│   ├── test_set/                   # Fixed test dataset
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   ├── new_data/                   # New data for incremental learning
│   ├── base.pt                     # Best trained model (mAP@0.5 > 0.90)
│   ├── train_incremental.py        # Main training pipeline
│   ├── incremental_pipeline.py     # Core pipeline functions
│   ├── test.py                     # Testing script
│   └── requirements.txt            # Python dependencies
│
├── YOLO-OCR-NLP/                   # 🚧 Coming soon
│   └── (to be added)
│
├── FastVLM/                        # 🚧 Coming soon
│   └── (to be added)
│
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🎯 Performance Summary

| Method | Speed | Accuracy (mAP@0.5) | Status |
|--------|-------|-------------------|--------|
| **YOLO-only** | ⚡ Millisecond-level | **> 0.90** | ✅ Implemented |
| YOLO + OCR + NLP | 🐢 Moderate-Slow | TBD | 🚧 Coming Soon |
| FastVLM | ⚡ Fast | TBD | 🚧 Coming Soon |

## 🚀 Quick Start (YOLO-only Pipeline)

### Prerequisites

- Python 3.12
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/yaoling000/Parking-sign-recognition.git
cd Parking-sign-recognition/YOLO-only

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

The repository includes a pre-trained model (`base.pt`) with **mAP@0.5 > 0.90** on the test set.

```bash
# Test the model
python test.py --weights base.pt --data test_set/data.yaml
```

### Incremental Learning Workflow

#### Stage 1: Auto-Labeling

```bash
python train_incremental.py \
    --stage 1 \
    --new_data new_data/ \
    --base_model base.pt \
    --output_dir output
```

This will:
- ✅ Scan for new images
- ✅ Remove duplicates (SHA-1 hashing)
- ✅ Generate automatic labels using base.pt
- ✅ Export to CVAT XML format

**Manual Step**: Review and correct labels in CVAT

#### Stage 2: Training

```bash
python train_incremental.py \
    --stage 2 \
    --output_dir output \
    --base_model base.pt \
    --test_set test_set \
    --epochs 50 \
    --batch 16
```

This will:
- ✅ Load corrected CVAT annotations
- ✅ Perform data augmentation
- ✅ Train new model version
- ✅ Validate on test set
- ✅ Compare with previous version

## 📊 YOLO-only Pipeline Features

### 1. Incremental Learning System
- Continuous model improvement with new data
- Automatic deduplication (SHA-1 hashing)
- Human-in-the-loop via CVAT

### 2. Intelligent Version Management
- Automatic versioning of models
- Performance tracking and comparison
- Easy rollback to previous versions

### 3. Smart Class Management
- Auto-merges new classes with existing ones
- Preserves class IDs across iterations
- Extracts class names from CVAT XML

### 4. High Performance
- **Current best model: mAP@0.5 > 0.90**
- Millisecond-level inference speed
- Optimized for known parking sign categories

## 📖 Detailed Documentation

For detailed usage of each pipeline, refer to the README files in each subdirectory.

## 🔬 Research Comparison

This project provides a comprehensive comparison of three different approaches:

- **Speed vs Accuracy**: YOLO-only achieves both high speed and accuracy for pre-seen categories
- **Generalization**: YOLO+OCR+NLP and FastVLM may perform better on unseen sign formats
- **Interpretability**: Different levels of explainability across methods

Full comparison results will be available after all three pipelines are implemented.

## 📝 Citation

```bibtex
@misc{parking-sign-recognition-2024,
  title={Australian Parking Sign Recognition: A Comparative Study},
  author={Your Team Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yaoling000/Parking-sign-recognition}}
}
```

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- CVAT annotation tool
- PaddleOCR (for OCR pipeline)
- Sydney street parking sign dataset

## 📧 Contact

- GitHub: [@yaoling000](https://github.com/yaoling000)
- Project Issues: [Report here](https://github.com/yaoling000/Parking-sign-recognition/issues)

---

**Current Status**: ✅ YOLO-only pipeline complete (mAP@0.5 > 0.90) | 🚧 Other pipelines coming soon
