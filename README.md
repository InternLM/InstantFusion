<p align="center">
  <img src="fig/instantfusion-banner.png" alt="InstantFusion — A Shared Latent Interface for Image Generators" width="100%">
</p>

<p align="center">
  <strong>Composing diffusion models through a shared latent space</strong>
</p>

<p align="center">
  <a href="https://github.com/InternLM/InstantFusion"><img src="https://img.shields.io/badge/Code-Available-2E7D32?style=flat-square&logo=github" alt="Code available"></a>
  <img src="https://img.shields.io/badge/Paper-Coming%20soon-9E9E9E?style=flat-square" alt="Paper coming soon">
  <a href="https://internlm.github.io/InstantFusion/"><img src="https://img.shields.io/badge/Project%20Page-GitHub%20Pages-2441DD?style=flat-square" alt="InstantFusion project page"></a>
  <a href="https://huggingface.co/XuanlangDai/InstantFusion"><img src="https://img.shields.io/badge/Model-Hugging%20Face-2441DD?style=flat-square&logo=huggingface" alt="Model checkpoints on Hugging Face"></a>
  <img src="https://img.shields.io/badge/Dataset-Coming%20soon-9E9E9E?style=flat-square" alt="Dataset coming soon">
</p>

<p align="center">
  <img src="fig/method.png" alt="InstantFusion method: shared-latent alignment, cross-model OPD, acceleration, and multi-reward composition" width="100%">
</p>
## Overview

InstantFusion maps the latents of different diffusion models into a shared latent space, so their denoising trajectories can be aligned and composed. 

**Latent alignment encoder (LAE).** Sigma-conditioned encoders and decoders map both models’ latents into the shared space. Training combines reconstruction, cross-reconstruction, latent alignment, and denoising-velocity alignment losses.

**Applications.** Once the models share a latent space, a trajectory can be handed off from one model to another mid-denoising. This enables:

- **Acceleration:** hand off between models with different step budgets.
- **Multi-preference composition:** combine models optimized for different preferences.
- **Cross-model OPD:** on-policy distillation of model B’s velocity field *v* from model A in the shared latent space.

This repository includes LAE training, OPD training, and LAE-based model handoff inference.

## Visual results

### LAE reconstruction quality

The examples below compare the baseline with the LAE reconstruction results for Qwen-Image, FLUX.1, and SD3. The figure also reports mean SSIM over 100 images for each model.

<p align="center">
  <a href="fig/reconstruct.pdf"><img src="fig/reconstruct.png" alt="LAE reconstruction examples and mean SSIM for Qwen-Image, FLUX.1, and SD3" width="100%"></a>
</p>
### Cross-model denoising handoff

These examples show the results of switching models and continuing denoising with the second model. The figure includes model pairings beyond the SD3/Qwen-Image inference implementation currently available in this repository.

<p align="center">
  <a href="fig/trans.pdf"><img src="fig/trans.png" alt="Image examples of cross-model handoff and continued denoising" width="100%"></a>
</p>
## Quick start

Trained InstantFusion checkpoints are available on [Hugging Face](https://huggingface.co/XuanlangDai/InstantFusion).

### 1. Set up the environment

```bash
git clone https://github.com/InternLM/InstantFusion.git
cd InstantFusion
conda create -n instantfusion python=3.10 -y
conda activate instantfusion
pip install -r requirements.txt
```

Use a compatible [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) checkout in the same environment. The training scripts can import it through `DIFFSYNTH_ROOT`:

```bash
export DIFFSYNTH_ROOT=/path/to/DiffSynth-Studio
```

Provide a Qwen-Image model directory containing `transformer/`, `text_encoder/`, `vae/`, and `tokenizer/`, plus an SD3 single-file checkpoint. Model weights are not bundled with this repository.

### 2. Prepare the dataset

> We use [VisPrompt5M](https://huggingface.co/datasets/CSU-JPG/VisPrompt5M) for training, which is not included in this repository.

The training data is supplied locally and is not included in this repository:

```text
DATASET_ROOT/
└── train/
    ├── metadata.csv
    └── images/
        └── example.png
```

`metadata.csv` needs `file_name` and `text` columns. Image paths in `file_name` are relative to `DATASET_ROOT/train/`:

```csv
file_name,text
images/example.png,"A lighthouse above a quiet sea at sunrise"
```

### 3. Train the latent alignment encoder

`--prepare-prompt-cache` creates the required SD3 and Qwen-Image prompt embeddings before training. Training uses `./prompt_cache` unless `--prompt-cache` is set. The cache is runtime data and should stay outside version control.

```bash
bash scripts/train_lae.sh \
  --qwen-path /path/to/Qwen-Image \
  --sd3-path /path/to/sd3.safetensors \
  --dataset-path /path/to/DATASET_ROOT \
  --output-path ./outputs/lae \
  --prepare-prompt-cache \
  --size 512 --precision bf16 \
  --steps-per-epoch 1000 --epochs 5
```

The LAE checkpoint stores the shared-latent bridge parameters.

### 4. Train cross-model OPD

Use the LAE checkpoint from the previous stage. OPD trains an SD3 LoRA with Qwen-Image as the frozen teacher. By default it uses 20 sampling steps and supervises zero-based steps `1,2,4,8`. Change these with `--sampling-steps` and `--supervised-step-ids` when needed.

```bash
bash scripts/train_opd.sh \
  --qwen-path /path/to/Qwen-Image \
  --sd3-path /path/to/sd3.safetensors \
  --bridge-checkpoint /path/to/lae.ckpt \
  --dataset-path /path/to/DATASET_ROOT \
  --output-path ./outputs/opd \
  --size 512 --precision bf16 \
  --steps-per-epoch 1000 --epochs 5
```

OPD reuses the LAE prompt cache by default. If you use a different dataset for OPD, add `--prepare-prompt-cache` to populate embeddings for its prompts.

### 5. Run LAE-based inference

```bash
bash scripts/infer.sh \
  --qwen-path /path/to/Qwen-Image \
  --sd3-path /path/to/sd3.safetensors \
  --lae-checkpoint /path/to/lae.ckpt \
  --prompt "A lighthouse above a quiet sea at sunrise" \
  --output-dir ./outputs/inference
```

### Advanced settings

The command-line interfaces expose paths and common controls. Put less frequently changed trainer or model settings in a JSON file and pass `--config path/to/config.json` to the relevant command above. JSON uses the Python setting names with underscores; model and dataset paths remain command-line arguments. For example, an OPD config can contain:

```json
{
  "learning_rate": 0.00001,
  "lora_rank": 64,
  "reference_weight": 0.1
}
```

To reuse the training prompt cache during inference, an inference config can contain:

```json
{
  "prompt_embedding_dir": "./prompt_cache",
  "sd3_cfg": 4.0,
  "qwen_cfg": 4.0
}
```

Explicit command-line options take precedence over JSON settings. Training commands also accept the previous underscore-style option names, though `--help` displays the shorter hyphen-style names. Run `bash scripts/train_lae.sh --help`, `bash scripts/train_opd.sh --help`, or `bash scripts/infer.sh --help` for the compact option lists.

## Repository layout

```text
fig/                     Method and result figures (PNG and source PDF)
scripts/train_lae.sh     LAE training entry point
scripts/train_opd.sh     OPD training entry point
scripts/infer.sh         LAE-based inference entry point
src/instantfusion/       Bridge, model-loading, cache, training, and inference modules
```

## Citation

The paper link and BibTeX entry will be added when the paper is available.
