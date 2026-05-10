# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gaussian Grouping extends 3D Gaussian Splatting to jointly reconstruct and segment objects in open-world scenes. Each Gaussian is augmented with a compact Identity Encoding supervised by SAM 2D masks and 3D spatial consistency regularization. The trained model supports scene editing: object removal, inpainting, colorization, and recomposition.

## Installation

Requires Python 3.12, PyTorch 2.11+, CUDA 12.8. See `INSTALL.md` for full details.

```bash
conda create -n gaussian_grouping python=3.12 && conda activate gaussian_grouping
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
```

## Common Commands

```bash
# Train on a dataset (runs train.py then render.py)
bash script/train.sh <dataset_name> <scale>
# e.g.: bash script/train.sh bear 1

# Prepare SAM/DEVA pseudo labels before training
bash script/prepare_pseudo_label.sh <dataset_name> <scale>

# Convert raw images to COLMAP sparse reconstruction
python convert.py -s <location>

# Render trained model
python render.py -m output/<dataset_name> --num_classes 256

# 3D object removal
bash script/edit_object_removal.sh output/<dataset_name> config/object_removal/<dataset>.json

# 3D object inpainting (run after removal)
bash script/edit_object_inpaint.sh output/<dataset_name> config/object_inpaint/<dataset>.json

# Evaluate metrics (PSNR, SSIM, LPIPS)
python metrics.py -m output/<dataset_name>

# Evaluate on LERF-Mask benchmark
python script/eval_lerf_mask.py <model_path>
```

## Architecture

### Core Pipeline

**Training** (`train.py`): Renders a random camera view each iteration, computes three losses, and performs Gaussian densification/pruning.

- **L1 + SSIM** on rendered RGB vs. ground truth
- **loss_obj**: Cross-entropy on projected 2D segmentation (classifier maps Identity Encodings → class logits)
- **loss_obj_3d** (every `reg3d_interval` steps): KL divergence enforcing nearby Gaussians to have similar Identity Encodings

**GaussianModel** (`scene/gaussian_model.py`): Stores per-Gaussian properties: `_xyz`, `_features_dc`/`_features_rest` (SH), `_scaling`, `_rotation`, `_opacity`, and crucially `_objects_dc` — the 16D Identity Encoding. The classifier is a `Conv2d(num_objects, num_classes, 1)` applied per-pixel on rendered object feature maps.

**Renderer** (`gaussian_renderer/__init__.py`): Thin wrapper around the CUDA `diff-gaussian-rasterization` submodule. Returns both the rendered RGB image and the rendered object feature map.

### Editing

**Object Removal** (`edit_object_removal.py`): Scores all Gaussians via the classifier, selects those above a threshold for the target object ID, expands the selection using a convex hull (Delaunay triangulation), then calls `removal_setup()` to filter them out.

**Inpainting** (`edit_object_inpaint.py`): After removal, fine-tunes remaining Gaussians using gradient masks (`inpaint_setup()` hooks zero gradients outside the inpaint region). New Gaussians are initialized via KD-tree interpolation from neighbors. Loss is masked L1 + LPIPS against 2D LaMa-inpainted images.

### Data Flow

```
data/<dataset>/
├── images/               # Training images
├── object_mask/          # Per-image SAM/DEVA segmentation masks
└── sparse/               # COLMAP reconstruction (cameras + points)

output/<dataset>/
├── point_cloud/iteration_<N>/point_cloud.ply   # Trained Gaussians
└── train/ test/          # Rendered images and metrics
```

### Key Parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 256 | Number of segmentation classes |
| `num_objects` | 16 | Identity Encoding dimension |
| `iterations` | 30,000 | Total training steps |
| `densify_from_iter` | 500 | Start densification |
| `densify_until_iter` | 15,000 | Stop densification |
| `reg3d_interval` | 2 | Apply 3D consistency loss every N steps |
| `reg3d_k` | 5 | Neighbors for 3D consistency |
| `reg3d_lambda_val` | 2 | 3D consistency loss weight |
| `lambda_dssim` | 0.2 | SSIM loss weight |

### CUDA Submodules

Both submodules required `#include <cstdint>` / `#include <cfloat>` patches for modern toolchains — these are already applied. Rebuild with `pip install --no-build-isolation submodules/<name>` if needed.
