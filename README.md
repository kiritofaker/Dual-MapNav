# Dual-MapNav

Implementation of **Dual-MapNav: Task-Driven Map Learning for Vision-Language Navigation** (arXiv 2026).

## Overview

Dual-MapNav proposes a novel approach to Vision-Language Navigation (VLN) by formulating map learning as a **conditional video generation** task. Using a pretrained Video Diffusion Model, Dual-MapNav generates future bird's-eye view (BEV) maps from current observations and text instructions.

## Key Features

- **Unified Map Representation**: Combines exploration nodes, semantic nodes, and text embeddings
- **Cross-Modal Fusion**: Integrates RGB, depth, and BEV observations
- **Conditional Video Diffusion**: Generates map trajectories via diffusion model
- **Two-Stage Training**: Pretraining + task-oriented fine-tuning
- **VLNTube Support**: Supports VLNTube/InteriorNav format data

## Project Structure

```
Dual-MapNav/
├── config/                 # Configuration files
├── models/                 # Core model implementations
│   ├── map_representation/ # Map encoding (BEV, Semantic, Text)
│   ├── observation/        # Observation encoding (RGB, Depth, Fusion)
│   ├── diffusion/         # Diffusion model (UNet3D, DDIM, CFG)
│   ├── map_predictor/      # Main Dual-MapNav model
│   └── trajectory/        # Trajectory generation
├── data/                  # Data loading (synthetic + VLN)
├── training/             # Training pipeline
├── inference/            # Inference and agent
├── utils/                # Utilities
└── scripts/              # Training/inference scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Synthetic Data Training

```bash
# Stage 1: Pretrain map generation
python scripts/train.py --stage 1 --epochs 100 --batch-size 8

# Stage 2: Task-oriented fine-tuning
python scripts/train.py --stage 2 --epochs 50 --batch-size 8
```

### VLNTube/InteriorNav Data Training

First, download the pre-built VLN data from [HuggingFace](https://huggingface.co/datasets/Eyz/CaffeEclipse):

```bash
# Then train with VLN data
python scripts/train_vln.py \
    --data-root /path/to/vlnverse \
    --split train \
    --instruction-type all \
    --stage 1 \
    --epochs 100 \
    --batch-size 4
```

### Inference

```bash
# Generate maps
python scripts/inference.py --num-samples 10

# Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/best.pt
```

### VLN Evaluation

```bash
# Evaluate with VLN data
python scripts/evaluate_vln.py \
    --data-root /path/to/vlnverse \
    --split val \
    --num-samples 100
```

## VLNTube Data Format

Dual-MapNav supports the VLNTube/InteriorNav format:

```
<data_root>/
├── <scene_id>/
│   └── <goal>_<start>/
│       ├── data/chunk-000/
│       │   └── episode_000000.parquet  # Positions, orientations, actions
│       ├── videos/chunk-000/
│       │   ├── observation.images.rgb/rgb.npy   # RGB sequence
│       │   └── observation.images.depth/depth.npy # Depth sequence
│       └── meta/
│           ├── episodes.jsonl
│           └── tasks.jsonl
```

### Instruction Types
- **fine-grained**: Per-trajectory image-based instructions (from Gemini)
- **coarse-grained**: Augmented goal instructions (template + caption fusion)
- **all**: Use both types

## Algorithm Overview

### 1. Map Representation

The map is represented with three components:
- **Exploration Nodes**: BEV features encoding traversable areas
- **Semantic Nodes**: Object semantic information
- **Text Embeddings**: Object category representations

### 2. Observation Encoding

RGB and depth observations are encoded using pretrained backbones, then fused with cross-modal attention.

### 3. Conditional Video Diffusion

Map prediction is formulated as video generation:
- The diffusion model predicts future map frames
- Conditioning on text instructions and current observations
- DDIM sampling for efficient generation

### 4. Trajectory Generation

Navigation trajectories are extracted via:
- A* path planning on predicted maps
- Video interpolation for smooth trajectories

## Citation

If you use this code, please cite:

```
@article{mapdream2026,
  title={Dual-MapNav: Task-Driven Map Learning for Vision-Language Navigation},
  author={},
  journal={arXiv},
  year={2026}
}
```
