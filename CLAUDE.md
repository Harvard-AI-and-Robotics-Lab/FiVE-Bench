# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

FiVE-Bench is a fine-grained video editing benchmark for evaluating diffusion and rectified flow models. The repository contains evaluation code, two video editing model implementations (Pyramid-Edit and Wan-Edit), and comprehensive metrics for assessing video editing quality.

## Architecture

### Core Components

- **Evaluation System** (`evaluation/`): Main evaluation logic with metrics calculation
  - `evaluate.py`: Primary evaluation script that orchestrates all metrics
  - `metrics_calculator.py`: Implements various video editing quality metrics
  
- **Video Editing Models** (`models/`):
  - `wan-edit/`: Rectified flow-based video editing implementation
    - `edit.py`: Main editing script for Wan-Edit model
    - `generate.py`: Video generation functionality
    - `wan/`: Core model architecture with modules for attention, VAE, text encoders
  - `pyramid-edit/`: Alternative editing method (implementation structure)

- **Configuration**: 
  - `config.yaml`: Central configuration for evaluation paths, metrics, and target methods
  - Must update `root_tgt_video_folder`, `cotracker_model_path`, and `IQA_PyTorch_model_path` with actual paths

### Data Structure

The benchmark expects data in `./data/` with:
- `videos/`: Source video files (.mp4)
- `images/`: Frame extracts organized by video ID
- `bmasks/`: Editing masks for localized evaluation
- `edit_prompt/`: JSON files with editing instructions (edit1-edit6_FiVE.json)
- `results/`: Target directory for edited video outputs

## Common Development Commands

### Environment Setup
```bash
conda create -n five-bench python=3.11.10 -y
conda activate five-bench
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.45.2
```

### Evaluation Commands

**Full evaluation (all metrics):**
```bash
sh scripts/eval_FiVE.sh
```

**FiVE-Acc only (VLM-based metric):**
```bash
sh scripts/eval_FiVE_acc_only.sh
```

**Custom evaluation with specific parameters:**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py --annotation_mapping_files "data/edit_prompt/edit5_FiVE.json" --tgt_methods "8_Wan_Edit"
```

### Video Editing (Wan-Edit)

**Run editing on FiVE dataset:**
```bash
sh models/wan-edit/scripts/run_FiVE.sh
```

**Single editing task:**
```bash
python models/wan-edit/edit.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 41 \
    --ckpt_dir hf/wan13/ \
    --data_dir data \
    --save_dir outputs \
    --FiVE_dataset_json data/edit_prompt/edit5_FiVE.json
```

### Installing Dependencies

**Core evaluation dependencies:**
```bash
pip install -r models/requirements.txt
```

**Co-Tracker (for Motion Fidelity Score):**
```bash
cd ../co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
```

**IQA-PyTorch (for NIQE metric):**
```bash
pip install pyiqa
# OR from source:
pip install git+https://github.com/chaofengc/IQA-PyTorch.git
```

## Key Configuration Requirements

Before running evaluation, update `config.yaml`:
- `cotracker_model_path`: Path to Co-Tracker model checkpoint
- `IQA_PyTorch_model_path`: Path to IQA-PyTorch installation  
- `root_tgt_video_folder`: Directory containing edited video results
- `tgt_methods`: List of method names to evaluate

## Evaluation Metrics

The benchmark implements comprehensive evaluation across:
- **Structure/Background Preservation**: PSNR, LPIPS, MSE, SSIM (masked regions)
- **Edit Consistency**: CLIP similarity between prompts and results
- **Image Quality**: NIQE scores
- **Temporal Consistency**: Motion Fidelity Score (MFS) via Co-Tracker
- **Edit Success**: FiVE-Acc using VLM-based question answering

## Directory Dependencies

The evaluation system expects this directory structure at the code level:
```
/path/to/code/
├── co-tracker/          # For MFS calculation
├── FiVE-Bench/         # This repository
└── IQA-PyTorch/        # For NIQE calculation
```