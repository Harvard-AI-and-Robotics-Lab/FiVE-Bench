# FiVE-Bench Video Editing Models

<img src="../assets/pyramid_edit_wan_edit.png" alt="rf-editing" width="700"/>

This directory contains two state-of-the-art video editing model implementations designed for the FiVE-Bench evaluation framework:

- **Pyramid-Edit**: A diffusion-based video editing method using the Pyramid-Flow architecture
- **Wan-Edit**: A rectified flow-based video editing approach leveraging the Wan2.1-T2V model

Both models support fine-grained video editing tasks including object transformations, style changes, background modifications, and temporal consistency preservation across 41-frame sequences.

## Environment Setup

Create and activate the conda environment:

```bash
conda create -n five-bench python=3.11.10 -y
conda activate five-bench
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.45.2
pip install -r models/requirements.txt
# Verify flash attention installation
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

---
# Pyramid-Edit

## Overview

Pyramid-Edit is a diffusion-based video editing method that leverages the Pyramid-Flow architecture for high-quality, temporally consistent video transformations. 

## Setup: Model Download
```bash
cd models/pyramid-edit
mkdir -p hf
cd hf

# Download [Pyramid-Flow](https://huggingface.co/rain1011/pyramid-flow-miniflux) model checkpoint
git clone https://huggingface.co/rain1011/pyramid-flow-miniflux
```

## Configuration

Before running Pyramid-Edit, update the configuration file `models/pyramid-edit/config.yaml`:

```yaml
device: 'cuda'
dtype: 'bf16'
model_name: 'pyramid_flux'  # or 'pyramid_mmdit'
model_path: 'models/pyramid-edit/hf/pyramid-flow-miniflux'
resolution: '384p'  # or '768p'
max_frames: 41
```

## Running Pyramid-Edit

### Single Video Editing

Edit a single video with custom prompts. This processes the bear example video, changing it from brown to purple.

```bash
# Run single video editing example
bash models/pyramid-edit/scripts/run_single.sh
```

### Running on FiVE Dataset

```bash
bash models/pyramid-edit/scripts/run_FiVE.sh
```

---
# Wan-Edit

## Overview

Wan-Edit is a rectified flow-based video editing method built upon the Wan2.1-T2V-1.3B model architecture. This approach provides efficient and high-quality video transformations through:

- **Rectified flow modeling**: Advanced flow-based generative approach for smoother video transitions
- **Text-to-video capabilities**: Strong text conditioning for precise edit control
- **1.3B parameter efficiency**: Optimized model size balancing performance and resource usage
- **832x480 resolution**: High-definition output suitable for detailed editing tasks

## Setup: Model Download
```bash
cd models/wan-edit
mkdir hf
# Download [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) model checkpoint to `models/wan-edit/hf/` directory
cd hf
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
```

## Running Wan-Edit on FiVE Dataset

To run Wan-Edit on the FiVE-Bench dataset:

```bash
# Run the complete FiVE evaluation script
bash models/wan-edit/scripts/run_FiVE.sh
```

This script will:
- Process all editing tasks (edit1 through edit6) in the FiVE-Bench dataset
- Use the Wan2.1-T2V-1.3B model with 832x480 resolution and 41 frames
- Require the model checkpoint in `models/wan-edit/hf/wan13/`
- Save results to `outputs/wan_edit_results/`

### Manual Execution

You can also run individual editing tasks manually:

```bash
sh models/wan-edit/scripts/run_single.sh
```

### Custom Parameters

For advanced usage, you can specify custom parameters:

```bash
python models/wan-edit/edit.py \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 41 \
    --ckpt_dir models/wan-edit/hf/Wan2.1-T2V-1.3B/ \
    --data_dir data \
    --save_dir outputs \
    --FiVE_dataset_json data/edit_prompt/edit5_FiVE.json
```

**Parameter Options:**
- `--task`: Model variant (t2v-1.3B)
- `--size`: Output resolution (832*480 recommended)
- `--frame_num`: Number of frames to generate (41 for FiVE-Bench)
- `--ckpt_dir`: Path to model checkpoint directory
- `--data_dir`: Input data directory
- `--save_dir`: Output directory for edited videos
- `--FiVE_dataset_json`: Specific editing task file

***Note:*** To specify a particular video, use the following arguments: 
```
--video_dir data/examples \
--video_name blackswan \
```

---

# Model Comparison & Selection

### When to Use Pyramid-Edit
- **High-quality requirements**: Better for applications requiring maximum visual fidelity
- **Flexible resolutions**: When you need both 384p and 768p output options

### When to Use Wan-Edit  
- **Efficiency focused**: Faster inference with 1.3B parameter model
- **Flow-based benefits**: Smoother temporal transitions and more stable generation
- **Text conditioning**: Superior text understanding for complex editing instructions
- **Resource constraints**: Better performance on limited computational resources

### Performance Characteristics (Wan-Edit > Pyramid-Edit)

| Aspect | Pyramid-Edit | Wan-Edit |
|--------|--------------|----------|
| **Model Size** | Larger (varies by variant) | 1.3B parameters |
| **Resolution** | 384p/768p | 832x480 |
| **Architecture** | Rectified flow | Rectified flow |
| **Inference Speed** | Slower | Faster |
| **Text Understanding** | Good | Excellent |
| **Memory Usage** | Higher | Lower |

### Performance Optimization

**1. GPU Memory Optimization**
- Use `bf16` precision instead of `fp32`
- Reduce `max_frames` if memory limited

**2. Inference Speed**
- Use single GPU with `CUDA_VISIBLE_DEVICES=0`
- Consider lower resolution for rapid prototyping