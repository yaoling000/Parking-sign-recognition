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
├── YOLO-only/                      # YOLO-only pipeline
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
├── YOLO-OCR-NLP/                   # YOLO + OCR + NLP pipeline (implemented)
│   ├── inference_det_v2/           # PaddleOCR detection model files
│   │   ├── inference/              # Model inference configuration
│   │   │   ├── inference.pdiparams      # Model parameters
│   │   │   ├── inference.pdiparams.info # Parameter metadata
│   │   │   └── inference.pdmodel        # Model architecture
│   │   └── ...
│   ├── eval_rules.py               # Parking rule evaluation and NLP parsing
│   ├── pipeline.py                 # Main pipeline: YOLO → OCR → NLP
│   ├── verify_install.py           # Environment verification script
│   ├── ground_truth.json           # Ground truth annotations for evaluation
│   ├── requirements.txt            # Python dependencies
│   └── yolo_best.pt                # Trained YOLOv8 model weights
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

## 🚀 Quick Start (YOLO-OCR-NLP Pipeline)

### Prerequisites

- Python 3.10+ (3.11 compatible)
- Windows (CPU), Linux (GPU optional), or macOS (minimal mode)
- ~5GB disk space (models + dependencies)
- Offline mode supported (FastVLM 1.5B)

### Installation

```bash
# Clone repository
git clone https://github.com/yuxuanma9-ar/Parking-sign-recognition.git
cd Parking-sign-recognition/YOLO-OCR-NLP

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows: .venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip wheel setuptools

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU-only, recommended for Windows)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

**Note for Windows users**: If you encounter `shm.dll` or `torch.dll` errors, install the Visual C++ Redistributable:
```
https://aka.ms/vs/17/release/vc_redist.x64.exe
```
Then restart your terminal and re-activate the virtual environment.

### Verify Installation

```bash
python verify_install.py
```

Expected output:
```
✓ NumPy OK
✓ OpenCV OK
✓ Albumentations OK
✓ PaddlePaddle OK
✓ PaddleOCR OK
✓ YOLOv8 OK
Passed 6/6 checks
```

### Run Complete Pipeline

```bash
# Run on all images in data/ folder
python pipeline.py

# Or specify custom input/output paths
python pipeline.py --input_dir data/ --output_dir results/
```

Expected output:
```
[INFO] Running pipeline on 10 images...
[INFO] Stage 1/3: YOLO Detection... ✓
[INFO] Stage 2/3: OCR Recognition... ✓
[INFO] Stage 3/3: NLP Parsing... ✓
[INFO] Total runtime: 92.58 seconds (9.25 sec/image)
[INFO] Results saved to: step_output/
```

Output structure:
```
step_output/
├── stepA_output/              # YOLO cropped sign regions + JSON
├── stepB_vis3.0/              # OCR visualizations overlaid on images
└── nlp_output3.0/             # Final structured parking rules (JSON)
    └── parking_rules_nlp_final_version.json
```

---

## 🧪 Quick Test

### Test on Sample Images

```bash
# Test with provided sample images
python pipeline.py --input_dir data/ --visualize

# Process a single image
python pipeline.py --input data/IMG_0001.JPG --output results/
```

### Evaluate Against Ground Truth

Compare your pipeline results against labeled annotations:

```bash
python eval_rules.py \
  --gt ground_truth.json \
  --pred nlp_output3.0/parking_rules_nlp_final_version.json \
  --ignore_nl \
  --gamma_fp_on_miss 0.30 \
  --fp_on_unmatched_pred 1
```

Expected output:
```
=== Evaluation Results ===
Precision: 0.892
Recall: 0.876
F1-Score: 0.884

Per-Component Scores:
  Days:     Precision=0.95, Recall=0.93, F1=0.94
  Times:    Precision=0.88, Recall=0.85, F1=0.87
  Duration: Precision=0.84, Recall=0.83, F1=0.84
```
## 🎯 Quick Examples

### Example 1: Basic Usage
```bash
python pipeline.py
```

### Example 2: Single Image with Visualization
```bash
python pipeline.py --input data/IMG_0001.JPG --visualize
```

### Example 3: Custom Confidence Threshold
```bash
python pipeline.py --conf_threshold 0.7 --ocr_threshold 0.8
```

### Example 4: Full Evaluation
```bash
python pipeline.py && \
python eval_rules.py \
  --gt ground_truth.json \
  --pred nlp_output3.0/parking_rules_nlp_final_version.json
```

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError: cv2 | `pip install opencv-python==4.6.0.66` |
| YOLO crashes on Windows | Install VC++ Redistributable |
| PaddleOCR installation fails | Use Python 3.10.x (not 3.12) |
| Empty OCR results | Check YOLO output in stepA_output/ |
| Low accuracy | Adjust thresholds or retrain models |

---

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

## 📊 YOLO + OCR + NLP Pipeline Features

### 1. High-Precision Sign Detection
- YOLOv8-based object detection fine-tuned on Australian parking signs  
- **mAP@0.5 > 0.90** on test dataset with diverse weather and lighting conditions  
- Robust to partial occlusions, angled views, and cluttered backgrounds  
- Outputs precise bounding boxes for downstream OCR processing  

### 2. Custom PaddleOCR with TTA Augmentation
- Fine-tuned detection model (`inference_det_v2/`) optimized for parking sign text  
- Test-Time Augmentation (TTA) with rotation, flipping, and scaling  
- Handles multi-oriented text, curved layouts, and small fonts  
- Advanced polygon filtering to remove noise and improve accuracy  

### 3. Rule-Based NLP Parser
- Converts raw OCR text into structured parking rules (JSON format)  
- Extracts time restrictions (e.g., "MON-FRI 8AM-6PM"), permit requirements, and special conditions  
- Handles complex multi-rule signs with arrows, exceptions, and zone indicators  
- Robust to OCR errors through fuzzy matching and domain-specific heuristics  

### 4. Modular Three-Stage Architecture
- **Stage A (YOLO)**: Sign localization and region extraction  
- **Stage B (OCR)**: Text recognition with visualization overlays  
- **Stage C (NLP)**: Rule parsing and structured output generation  
- Each stage produces intermediate outputs for debugging and quality assurance  

### 5. CPU-Optimized for Production
- Fully functional on **CPU-only** systems (no GPU required)  
- Optimized inference pipeline: **~9 seconds per image** on standard hardware  
- Lightweight dependencies with minimal installation footprint  
- Windows, Linux, and macOS compatible with unified codebase  

### 6. Comprehensive Evaluation Framework
- Ground truth comparison with precision, recall, and F1-score metrics  
- Per-component evaluation (days, times, duration)  
- Configurable false-positive penalties and matching strategies  
- Automated batch evaluation with detailed error analysis  

### 7. Explainable & Debuggable Pipeline
- Visual outputs at each stage for human verification  
- JSON outputs retain original OCR text alongside parsed rules  
- Clear separation between detection, recognition, and parsing errors  
- Detailed logging and error tracking for failure diagnosis  

### 8. Flexible Input/Output Handling
- Batch processing of multiple images with parallel inference  
- Configurable confidence thresholds for YOLO and OCR  
- Multiple output formats: JSON, visualization images, and logs  
- Easy integration with external systems via standardized JSON schema  

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
