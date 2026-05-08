# Installation

Tested with Python 3.12, PyTorch 2.11, CUDA 12.8, RTX 3070.

## 1. Create environment

```bash
conda create -n gaussian_grouping python=3.12 -y
conda activate gaussian_grouping
```

## 2. Install PyTorch

Install a PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install dependencies

```bash
pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips
```

## 4. Install CUDA submodules

Both submodules require `--no-build-isolation` so that pip's build subprocess can access the installed torch.

```bash
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
```

## Notes

- The original project targeted Python 3.8 / PyTorch 1.12 / CUDA 11.3. Two source-level fixes were needed to build against modern toolchains:
  - `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` — added `#include <cstdint>` for `uint32_t` / `uint64_t` / `std::uintptr_t`.
  - `submodules/simple-knn/simple_knn.cu` — added `#include <cfloat>` for `FLT_MAX`.
- `ninja` is optional but speeds up compilation significantly (`pip install ninja`).
