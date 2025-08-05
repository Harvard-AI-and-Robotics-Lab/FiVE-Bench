# Installation Guide

## Table of Contents
- [Step 1: Create Conda Environment](#step-1-create-conda-environment)
- [Step 2: Install FiVE-Bench and Dependencies](#step-2-install-five-bench-and-dependencies)
  - [Clone FiVE-Bench Repository](#clone-five-bench-repository)
  - [Install Co-Tracker and IQA Repos](#install-co-tracker-and-iqa-repos)
- [Step 3: Run FiVE-Bench Evaluation](#step-3-run-five-bench-evaluation)
  - [Evaluation Example: Wan-Edit](#evaluation-example-wan-edit)
  - [Evaluate Your Own Method](#evaluate-your-own-method)



---
## Step 1: Create Conda Environment

```bash
conda create -n five-bench python=3.11 -y
conda activate five-bench
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Step 2: Install FiVE-Bench and Dependencies

⭐ After installation, your directory structure should look like this:

```
📁 /path/to/code
├── 📁 co-tracker
├── 📁 FiVE-Bench
├── 📁 IQA-PyTorch
```
Make sure all dependencies for each subproject are installed accordingly.

> ⚠️ **NOTE:** Replace `/path/to/code` in the [`./config.yaml`](./config.yaml) file with the actual path to your ***code*** directory.

### ⬇️ Install Co-Tracker and IQA Repos
- Motion Fidelity Score (MFS) @ Co-Tracker: To evaluate temporal consistency using MFS, install [Co-Tracker](https://github.com/facebookresearch/co-tracker) in the following path: `./code/co-tracker`.
    ```bash
    cd ./code
    git clone https://github.com/facebookresearch/co-tracker
    cd co-tracker
    pip install -e .
    pip install matplotlib flow_vis tqdm tensorboard


    mkdir -p checkpoints
    cd checkpoints
    # download the offline (single window) model
    wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
    cd ..
    ```


- Image Quality Assessment (IQA) @ NIQE: To evaluate image quality with NIQE, install [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) under `./code/IQA-PyTorch`.
Then, replace the default `inference_iqa.py` with the version provided in our repo at [`./files/inference_iqa.py`](./files/inference_iqa.py).

    ```bash
    # Install with pip
    pip install pyiqa

    # Install latest github version
    pip uninstall pyiqa # if have older version installed already 
    pip install git+https://github.com/chaofengc/IQA-PyTorch.git

    # Install with git clone
    cd ./code
    git clone https://github.com/chaofengc/IQA-PyTorch.git
    cd IQA-PyTorch
    # pip install -r requirements.txt
    python setup.py develop
    ```

    💡 Don’t forget to replace `inference_iqa.py`:
    ```bash
    cp ../../files/inference_iqa.py ./inference_iqa.py
    ```

### ⬇️ Clone FiVE-Bench Repository
Download dataset and install the evaluation code

```bash
cd ./code
# evaluation code
git clone https://github.com/minghanli/FiVE-Bench.git
pip install -r requirements.txt

# FiVE-Bench dataset 
cd ./FiVE-Bench
git clone https://huggingface.co/datasets/LIMinghan/FiVE-Fine-Grained-Video-Editing-Benchmark
mv FiVE-Fine-Grained-Video-Editing-Benchmark data
unzip bmasks.zip images.zip videos.zip
```

The data structure should looks like:

  ```json
  📁 data
  ├── 📁 assets/
  ├── 📁 edit_prompt/
  │   ├── 📄 edit1_FiVE.json
  │   ├── 📄 edit2_FiVE.json
  │   ├── 📄 edit3_FiVE.json
  │   ├── 📄 edit4_FiVE.json
  │   ├── 📄 edit5_FiVE.json
  │   └── 📄 edit6_FiVE.json
  ├── 📄 README.md
  ├── 📦 bmasks.zip 
  ├── 📁 bmasks 
  │   ├── 📁 0001_bus
  │       ├── 🖼️ 00001.jpg
  │       ├── 🖼️ 00002.jpg
  │       ├── 🖼️ ...
  │   ├── 📁 ...
  ├── 📦 images.zip 
  ├── 📁 images
  │   ├── 📁 0001_bus
  │       ├── 🖼️ 00001.jpg
  │       ├── 🖼️ 00002.jpg
  │       ├── 🖼️ ...
  │   ├── 📁 ...
  ├── 📦 videos.zip 
  ├── 📁 videos
  │   ├── 🎞️ 0001_bus.mp4
  │   ├── 🎞️ 0002_girl-dog.mp4
  │   ├── 🎞️ ...
  ```

---

## Step 3: Run FiVE-Bench Evaluation

### 🎯 Evaluation Example: Wan-Edit
As an example, you can run evaluation using the **Wan-Edit** results. We use the edited results in `./data/results/Wan-Edit` with prompts from `./data/edit_prompt/edit5_FiVE.json`. Then run:

```bash
cd FiVE-Bench
sh scripts/eval_FiVE.sh --annotation_mapping_files "data/edit_prompt/edit5_FiVE.json" --tgt_methods "8_Wan_Edit" 
```

The evaluation result files should be found in:


```
📁 outputs
├── 📄 edit5_FiVE_evaluation_result_frame_stride8.csv
├── 📄 edit5_FiVE_evaluation_result_frame_stride8_avg.csv
```

### 🎯 Evaluate Your Own Method
If you want to evaluate **your own method**, you can modify the following parameters in [`config.yaml`](./config.yaml) and [`evaluation/evaluate.py`](evaluation/evaluate.py):

- `root_tgt_video_folder`: the root directory where your edited videos are stored  
- `all_tgt_video_folders`: a list of subfolders corresponding to your method(s)

Updating these paths allows the evaluation script to locate and assess your results accordingly.

---