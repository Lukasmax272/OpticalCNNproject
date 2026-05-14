# OpticalCNNproject
Lukas' optical CNN python simulation
This repository contains a differentiable simulation of Optical Neural Networks (ONNs) for MNIST classification. The project follows a progression from simple diffractive layers (D²NN) to translation-invariant hybrid architectures (ODCNN).

## Project Architecture

The codebase is organized to show the evolution of optical architectures:

### Core Simulation
- **[physics_sim.py](file:///c:/CodeProjects/Optical_rec/physics_sim.py)**: Implements the Angular Spectrum Method (ASM) for free-space propagation. It defines the `OpticalSimulator` and `MultiLayerD2NN` classes.
- **[four_f_sim.py](file:///c:/CodeProjects/Optical_rec/four_f_sim.py)**: Implements the 4f correlator setup using FFTs and the `MotionPoolingFourierMask` for translation robustness.
- **[four_f_hybrid.py](file:///c:/CodeProjects/Optical_rec/four_f_hybrid.py)**: Contains the `ODCNNHybrid` model, combining a tiled 4f correlator with a multi-layer D²NN classifier.
- **[utils.py](file:///c:/CodeProjects/Optical_rec/utils.py)**: Utilities for data loading (MNIST), phase-encoding, and shift augmentation.

### Training & Entry Points
- **[main.py](file:///c:/CodeProjects/Optical_rec/main.py)**: Training for Phase 1 (Single and Multi-layer D²NN).
- **[main_phase2.py](file:///c:/CodeProjects/Optical_rec/main_phase2.py)**: Training for Phase 2a (Naive 4f with `amax` pooling).
- **[main_phase2_motion_pool.py](file:///c:/CodeProjects/Optical_rec/main_phase2_motion_pool.py)**: Training for Phase 2b (4f with Motion Pooling).
- **[main_phase2_hybrid.py](file:///c:/CodeProjects/Optical_rec/main_phase2_hybrid.py)**: Training for Phase 2c (ODCNN Hybrid).

### Evaluation & Analysis
- **[benchmark.py](file:///c:/CodeProjects/Optical_rec/benchmark.py)**: The primary evaluation harness. Use `--plot-all` to generate the comparison of all architectures across a shift sweep.
- **[plot_all_training_curves.py](file:///c:/CodeProjects/Optical_rec/plot_all_training_curves.py)**: Aggregates history files into a single training visualization.
- **[visualize_full_process.py](file:///c:/CodeProjects/Optical_rec/visualize_full_process.py)**: Visualizes the propagation of a single digit through the optical system.

## Directory Map
- **Root Directory**: Contains primary entry points (`main*.py`), core physics (`physics_sim.py`), and legacy results/plots (`.npy`, `.png`).
- `data/`: Local cache of the MNIST dataset.
- `results/`: Standardized location for newer checkpoints (e.g., `odcnn_hybrid.pt`) and benchmark JSONs.
- `images/`: Curated visualizations and GIFs (some plots are also in the root).
- `browser/`: A web-based "Wave Explorer" for interactive visualization of results.
- `scratch/`: Experimental scripts, temporary tests, and diagnostic tools.

## How to Run

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Training
To train the Phase 2b model (Motion Pooling):
```bash
python main_phase2_motion_pool.py --grid 64 --iters 5000
```

### 3. Benchmarking
To compare all trained variants:
```bash
python benchmark.py --plot-all
```
