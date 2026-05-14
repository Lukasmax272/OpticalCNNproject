import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from physics_sim import MultiLayerD2NN
from utils import load_mnist_generator
from main import build_bin_masks, GRID, DEVICE

masks = np.load("learned_phase_mask.npy")
num_layers = masks.shape[0]

z_total = 0.096
spacings = [z_total / (num_layers + 1)] * (num_layers + 1)
sim = MultiLayerD2NN(
    num_layers=num_layers,
    grid_size=GRID,
    layer_spacings=spacings,
    device=DEVICE
)

for i in range(num_layers):
    with torch.no_grad():
        sim.phase_masks[i].copy_(torch.from_numpy(masks[i]).to(DEVICE))

gen = load_mnist_generator(1, DEVICE, train=False)
field, labels = next(gen)
label = labels[0].item()

bin_masks = build_bin_masks(64, DEVICE).cpu().numpy()

with torch.no_grad():
    input_phase = torch.angle(field[0]).cpu().numpy()
    intensity = sim(field)[0].cpu().numpy()

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.patch.set_facecolor('#1e1e1e')
for ax in axes:
    ax.set_facecolor('#1e1e1e')

fig.suptitle(f"Phase 1b 3-Layer D²NN Forward Pass - Digit: {label}", fontsize=20, color='#f8f8f2', fontweight='bold')

im0 = axes[0].imshow(input_phase, cmap="inferno")
axes[0].set_title("Input Field Phase", color='#f8f8f2', fontsize=16, pad=10)
axes[0].axis('off')

# Masks
for i in range(num_layers):
    mask_wrapped = np.mod(masks[i], 2 * math.pi)
    im = axes[i+1].imshow(mask_wrapped, cmap="twilight", vmin=0, vmax=2*math.pi)
    axes[i+1].set_title(f"Layer {i+1} Phase Mask", color='#f8f8f2', fontsize=16, pad=10)
    axes[i+1].axis('off')

# Intensity with bins
axes[4].imshow(intensity, cmap="hot")
axes[4].set_title("Detector Intensity", color='#f8f8f2', fontsize=16, pad=10)
axes[4].axis('off')

colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    y, x = np.where(bin_masks[i] > 0)
    if len(y) > 0:
        min_y, max_y = y.min(), y.max()
        min_x, max_x = x.min(), x.max()
        # Correct label is solid, others dashed
        rect = plt.Rectangle((min_x - 0.5, min_y - 0.5), max_x - min_x + 1, max_y - min_y + 1, 
                             linewidth=2.5 if i == label else 1.5, 
                             edgecolor='#50fa7b' if i == label else '#ff5555', 
                             facecolor='none',
                             linestyle='-' if i == label else '--')
        axes[4].add_patch(rect)
        axes[4].text(min_x, min_y - 2, str(i), 
                     color='#50fa7b' if i == label else '#ff5555', 
                     fontsize=14, fontweight='bold')

plt.tight_layout()
plot_path = "3_layer_visualization.png"
plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"Saved {plot_path}")
