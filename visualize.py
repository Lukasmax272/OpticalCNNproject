"""Visualize the learned phase mask and its diffraction pattern."""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from physics_sim import OpticalSimulator
from utils import load_mnist_generator
from main import build_bin_masks

DEVICE = torch.device("cpu")

mask = np.load("learned_phase_mask.npy")
phase_mask = torch.from_numpy(mask).to(DEVICE)

sim = OpticalSimulator(size=64, device=DEVICE)
gen = load_mnist_generator(64, DEVICE, train=False)
bin_masks = build_bin_masks(64, DEVICE).cpu().numpy()

# Wrap to [0, 2π] for visualization (training is unbounded but display is cyclic).
mask_wrapped = np.mod(mask, 2 * math.pi)

# Get one batch of intensity through the trained mask.
field, labels = next(gen)
with torch.no_grad():
    intensity = sim.propagate(field, phase_mask)[0].cpu().numpy()  # one digit

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mask_wrapped, cmap="twilight")
axes[0].set_title("Learned phase mask (wrapped to [0, 2π])")

axes[1].imshow(intensity, cmap="hot")
axes[1].set_title(f"Intensity at detector (input: '{labels[0].item()}')")

axes[2].imshow(intensity, cmap="hot")
axes[2].imshow(bin_masks.sum(0), cmap="cool", alpha=0.3)
axes[2].set_title("Intensity + bin overlay")

plt.tight_layout()
plt.savefig("phase1_results.png", dpi=120)
plt.show()
print("Saved phase1_results.png")
