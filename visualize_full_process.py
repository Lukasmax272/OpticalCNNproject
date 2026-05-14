"""Visualize the full D²NN propagation process."""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from physics_sim import OpticalSimulator
from utils import load_mnist_generator
from main import build_bin_masks

DEVICE = torch.device("cpu")

# Load simulator
sim = OpticalSimulator(size=64, device=DEVICE)

# Try loading a learned mask, otherwise fall back to a dummy mask
if os.path.exists("learned_phase_mask.npy"):
    mask = np.load("learned_phase_mask.npy")
    print("Using learned_phase_mask.npy")
else:
    mask = np.random.rand(64, 64) * 2 * math.pi
    print("Using random phase mask (learned mask not found)")

phase_mask = torch.from_numpy(mask).to(DEVICE)
mask_wrapped = np.mod(mask, 2 * math.pi)

# Get one batch of input fields
gen = load_mnist_generator(1, DEVICE, train=False)
field, labels = next(gen)
label = labels[0].item()

bin_masks = build_bin_masks(64, DEVICE).cpu().numpy()

# Forward pass step-by-step
with torch.no_grad():
    input_phase = torch.angle(field[0]).cpu().numpy()
    
    # 1. Apply SLM mask (at z = 0)
    modulated_field = sim._apply_slm(field, phase_mask)
    modulated_phase = torch.angle(modulated_field[0]).cpu().numpy()
    
    # 2. Propagate through free space to detector plane (at z = 9.6 cm)
    intensity = sim.propagate(field, phase_mask)[0].cpu().numpy()

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"Phase 1 D²NN Optical Forward Pass - Digit: {label}  (Distance: {sim.z*100:.1f} cm)", fontsize=18)

# 1. Input Image Phase
im0 = axes[0].imshow(input_phase, cmap="inferno")
axes[0].set_title("1. Input Field Phase\n(Digit, z = 0 cm)")
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# 2. Learned Phase Mask
im1 = axes[1].imshow(mask_wrapped, cmap="twilight", vmin=0, vmax=2*math.pi)
axes[1].set_title("2. SLM Phase Mask\n(Mask, z = 0 cm)")
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# 3. Output Intensity
im2 = axes[2].imshow(intensity, cmap="hot")
axes[2].set_title(f"3. Output Intensity\n(Detector, z = {sim.z*100:.1f} cm)")
axes[2].axis('off')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# 4. Output with Bins
axes[3].imshow(intensity, cmap="hot")
axes[3].set_title("4. Output with Detector Bins")
axes[3].axis('off')

# Overlay bins cleanly
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    y, x = np.where(bin_masks[i] > 0)
    if len(y) > 0:
        min_y, max_y = y.min(), y.max()
        min_x, max_x = x.min(), x.max()
        rect = plt.Rectangle((min_x - 0.5, min_y - 0.5), max_x - min_x + 1, max_y - min_y + 1, 
                             linewidth=2, edgecolor=colors[i], facecolor='none',
                             linestyle='--' if i != label else '-')
        axes[3].add_patch(rect)
        axes[3].text(min_x, min_y - 2, str(i), color=colors[i], fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("full_process_visualization.png", dpi=120)
print("Saved full_process_visualization.png")
