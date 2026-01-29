# SGNO

This repository provides a minimal PyTorch implementation of the Spectral Generator Neural Operator (SGNO) for one step prediction and autoregressive rollouts.

The code is self contained and includes:
- SGNO model definitions for 1D, 2D, and 3D
- A wrapper that accepts tensors in common trajectory layouts
- Lightweight scripts for sanity checking, training, and evaluation on NPZ trajectory files

## Installation

Python 3.10 or newer is recommended.

Create an environment and install dependencies:
```
pip install -r requirements.txt
```

Optional developer dependencies:
```
pip install -r requirements_dev.txt
```

## Quick sanity check

Run a forward pass for 1D, 2D, and 3D:
```
python scripts/sanity_check.py
```

## NPZ dataset format

The training and evaluation scripts expect an NPZ file containing one or both of the following arrays:
- train: float32 array with shape (N, T, C, X) for 1D, (N, T, C, X, Y) for 2D, or (N, T, C, X, Y, Z) for 3D
- test: same layout as train

If only u is present, the scripts will use it and split it into train and test.

## Train

Example using a JSON config:
```
python scripts/train_npz.py --config configs/example_2d.json --data /path/to/data.npz --out runs/example_2d
```

## Evaluate rollouts

```
python scripts/eval_npz.py --config configs/example_2d.json --data /path/to/data.npz --ckpt runs/example_2d/best.pt --out runs/example_2d/eval.json
```

## Model configuration

The model is constructed from a network_config string. The example configs use the key value form:
```
sgno;width=20;modes=8;n_blocks=4;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=smooth;filter_strength=1.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0
```

The build entry point is:
```
from sgno import build_sgno_from_config
model = build_sgno_from_config(network_config, num_spatial_dims, num_points, num_channels)
```
