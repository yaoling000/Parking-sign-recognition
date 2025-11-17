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

2. **YOLO + OCR + NLP Pipeline** ✅ (Current Implementation)
   - YOLO for sign and symbol detection
   - PaddleOCR for text extraction
   - NLP for converting text into structured output

3. **FastVLM Pipeline** ✅ (Current Implementation)
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
├── FastVLM/
│   │── sft_data/                   # Supervised Fine-Tuning (SFT) dataset
│    ├── train.json             # Training split (image–JSON instruction pairs)
│    └── val.json               # Validation split for SFT
│── pref_data/                  # Direct Preference Optimization (DPO) data
│    └── dpo_pairs_v2.jsonl     # Chosen vs rejected pairs for preference learning
│── gt_json/                    # Ground-truth labels for evaluation
│    ├── IMG_xxxx.json          # Parking sign rule annotations
│    └── ...                    # (Used for scoring metrics)
│── images/                     # Raw parking sign images used for evaluation
│    └── *.JPG / *.png          # Input images for SFT, DPO, and inference tests
│── preds_sft/                  # Predictions from SFT-only model
│    └── *.json                 # Model outputs before preference optimization
│── preds_dpo/                  # Predictions from DPO-optimized model
│    └── *.json                 # Final JSON outputs (better structure & accuracy)
│── preds_timed/                # Time-measured prediction results
│    └── IMG_xxx.json           # Used for runtime analysis (per-image latency)
│── dpo_candidates/             # Candidate outputs generated during pair creation
│    └── sample_chosen.json     # “Chosen” response examples
│    └── sample_rejected.json   # “Rejected” response examples
│── ml-fastvlm/                 # Base FastVLM 1.5B model (local checkpoint)
│    └── checkpoints/           # Contains vision tower + language model weights
│                               # (Ignored by .gitignore due to large size)
│── tools/                      # Utility scripts
│    └── eval_score.py          # Scoring script (precision, recall, JSON correctness)
│    └── merge_lora.py          # Merge LoRA adapters into the base model
│    └── dpo_data_builder.py    # Script for generating preference pairs
│── FastVLM+RLHF.ipynb          # Main training & inference notebook
│                               # - Loads FastVLM offline
│                               # - Runs SFT (LoRA)
│                               # - Runs DPO optimization
│                               # - Performs evaluation & timing tests
└── requirements.txt            # Python dependencies (transformers, peft, torch, etc.)
│
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🎯 Performance Summary

| Method | Speed | Accuracy | Status |
|--------|-------|-------------------|--------|
| **YOLO-only** | ⚡ Millisecond-level | **> 0.90** | ✅ Implemented |
| **YOLO + OCR + NLP** | 🐢 Moderate-Slow | **> 0.80** | ✅ Implemented |
| **FastVLM** | ⚡ Fast | **> 0.30** | ✅ Implemented |

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

## 🚀 Quick Start (FastVLM + RLHF Pipeline)

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~15GB disk space (model + LoRA)
- Offline mode supported (FastVLM 1.5B)

### Installation

```bash
# Clone repository
git clone https://github.com/yaoling000/Parking-sign-recognition.git
cd Parking-sign-recognition/FastVLM_RLHF

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```bash
python tools/eval_score.py --pred_dir preds_sft/ --gt_dir gt_json/
```

This will:
- ✅ Load the SFT-trained FastVLM model  
- ✅ Parse parking-rule information from images  
- ✅ Compare predictions with ground-truth JSONs  
- ✅ Save results to `preds_sft/`

---

### RLHF Workflow

#### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python FastVLM_SFT.py \
    --train_json sft_data/train.json \
    --val_json sft_data/val.json \
    --output_dir outputs/sft_lora/ \
    --epochs 3 \
    --lora_rank 8
```

This will:
- ✅ Load FastVLM 1.5B in offline mode  
- ✅ Train LoRA adapters using labeled image–JSON pairs  
- ✅ Improve rule interpretation and JSON structure quality  
- ✅ Save LoRA weights into `outputs/sft_lora/`

**Manual Step**: Review SFT predictions in `preds_sft/`.

---

#### Stage 2: Direct Preference Optimization (DPO)

```bash
python FastVLM_DPO.py \
    --pair_file pref_data/dpo_pairs_v2.jsonl \
    --sft_lora outputs/sft_lora/ \
    --output_dir outputs/dpo_lora/ \
    --epochs 2
```

This will:
- ✅ Load the SFT-trained LoRA  
- ✅ Train using chosen vs rejected preference pairs  
- ✅ Improve consistency and structure of generated rules  
- ✅ Save DPO LoRA weights into `outputs/dpo_lora/`

---

### Final Prediction Test

```bash
python tools/eval_score.py \
    --pred_dir preds_dpo/ \
    --gt_dir gt_json/
```

This will:
- ✅ Evaluate the DPO-optimized model  
- ✅ Output precision, recall, and JSON structure

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

## 📊 FastVLM + RLHF Pipeline Features

### 1. Instruction-Following Understanding
- Extracts complex parking rules from both text and symbols  
- Handles multi-line layouts, arrows, time ranges, and exceptions  
- Robust to sign damage, shadows, and partial occlusions  

### 2. Supervised Fine-Tuning (SFT)
- Aligns model outputs with curated image–JSON pairs  
- Corrects structure inconsistencies in generated rules  
- Produces stable and predictable machine-readable outputs  

### 3. Direct Preference Optimization (DPO)
- Learns human-like preferences between “better” vs “worse” answers  
- Improves reasoning traces and reduces invalid outputs  
- Enhances logical consistency across multi-rule signs  

### 4. Offline & Lightweight LoRA Training
- Fully offline pipeline (no external API required)  
- LoRA updates train only **0.55%** of model weights  
- Efficient training on consumer GPUs  

### 5. High-Quality Structured Output
- Generates standardized JSON with rule names, time windows, and arrows  
- Ensures consistent field formatting across images  
- Suitable for downstream parsing or constraint validation  

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
@misc{parking-sign-recognition-2025,
  title={Australian Parking Sign Recognition: A Comparative Study},
  author={CS15},
  year={2025},
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
