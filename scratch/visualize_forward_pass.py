import sys
sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from physics_sim import MultiLayerD2NN
from four_f_sim import MotionPoolingFourierMask
from utils import load_mnist_generator
from main import build_bin_masks

# Setup
os.makedirs('images', exist_ok=True)
DEVICE = torch.device('cpu')
GRID = 64

# Load models
z_total = 0.096
# Phase 1 (1L)
mask_1l_raw = np.load("learned_phase_mask_1l_64p.npy")[0]
spacings_1l = [z_total/2, z_total/2]
sim_p1 = MultiLayerD2NN(num_layers=1, grid_size=GRID, layer_spacings=spacings_1l, device=DEVICE)
sim_p1.phase_masks[0].data.copy_(torch.from_numpy(mask_1l_raw))

# Phase 1b (3L)
mask_3l_raw = np.load("learned_phase_mask_3l_64p.npy")
spacings_3l = [z_total/4] * 4
sim_3l = MultiLayerD2NN(num_layers=3, grid_size=GRID, layer_spacings=spacings_3l, device=DEVICE)
for i in range(3):
    sim_3l.phase_masks[i].data.copy_(torch.from_numpy(mask_3l_raw[i]))

# Phase 2b
mask_p2b_raw = np.load("learned_phase_mask_4f_motion_64.npy")
sim_p2b = MotionPoolingFourierMask(size=GRID, device=DEVICE)
H_p2b = torch.from_numpy(mask_p2b_raw)

# Bin masks for accuracy check
bin_masks = build_bin_masks(GRID, DEVICE)

# Find a digit all three get correct
print("Finding a correctly classified digit...")
gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
while True:
    field, labels = next(gen)
    label = labels[0].item()
    
    with torch.no_grad():
        # P1-1L
        i1 = sim_p1(field)
        b1 = torch.einsum("bhw,khw->bk", i1, bin_masks)
        p1_ok = b1.argmax(-1).item() == label
        
        # P1-3L
        i3 = sim_3l(field)
        b3 = torch.einsum("bhw,khw->bk", i3, bin_masks)
        p3_ok = b3.argmax(-1).item() == label
        
        # P2b
        i2b = sim_p2b.propagate(field, H_p2b)
        b2b = torch.einsum("bhw,khw->bk", i2b, bin_masks)
        p2b_ok = b2b.argmax(-1).item() == label
        
        if p1_ok and p3_ok and p2b_ok:
            print(f"Found digit {label} correct for all!")
            intensity_p1 = i1[0].numpy()
            intensity_3l = i3[0].numpy()
            intensity_p2b = i2b[0].numpy()
            # Also get raw 2b
            spec = torch.fft.fft2(field)
            mod = spec * H_p2b
            out = torch.fft.ifft2(mod)
            intensity_p2b_no_pool = (out.real.square() + out.imag.square())[0].numpy()
            break

digit_img = np.mod(torch.angle(field[0]).numpy(), 2*np.pi)

# Visualization
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#121212')

def plot_intensity(ax, data, title):
    # Log scale: log1p compresses the dynamic range so dimmer features appear
    im = ax.imshow(np.log1p(data), cmap='magma')
    ax.set_title(title, color='white', fontsize=14)
    ax.axis('off')
    return im

# Row 1: Phase 1 (1-Layer)
ax1 = plt.subplot2grid((3, 4), (0, 0))
ax1.imshow(digit_img, cmap='gray')
ax1.set_title(f"Input Digit: {label}", color='white', fontsize=14)
ax1.axis('off')

ax2 = plt.subplot2grid((3, 4), (0, 1))
p1_wrapped = np.mod(mask_1l_raw, 2*np.pi)
im2 = ax2.imshow(p1_wrapped, cmap='twilight', vmin=0, vmax=2*np.pi)
ax2.set_title("P1-1L Mask", color='white', fontsize=14)
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = plt.subplot2grid((3, 4), (0, 2))
plot_intensity(ax3, intensity_p1, "P1-1L Result (Log)")

# Row 2: Phase 1b (3-Layer)
for i in range(3):
    ax = plt.subplot2grid((3, 4), (1, i))
    m_wrap = np.mod(mask_3l_raw[i], 2*np.pi)
    im = ax.imshow(m_wrap, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax.set_title(f"P1-3L Layer {i+1}", color='white', fontsize=14)
    ax.axis('off')
    if i == 2: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax_res3 = plt.subplot2grid((3, 4), (1, 3))
plot_intensity(ax_res3, intensity_3l, "P1-3L Result (Log)")

# Row 3: Phase 2b (4f)
ax4 = plt.subplot2grid((3, 4), (2, 1))
H_shifted = np.fft.fftshift(mask_p2b_raw)
p2b_phase = np.angle(H_shifted)
im4 = ax4.imshow(p2b_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
ax4.set_title("P2b Fourier Phase", color='white', fontsize=14)
ax4.axis('off')
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

ax5 = plt.subplot2grid((3, 4), (2, 0))
impulse = torch.fft.fftshift(torch.fft.ifft2(torch.from_numpy(mask_p2b_raw))).real.numpy()
im5 = ax5.imshow(impulse, cmap='bwr', vmin=-np.max(np.abs(impulse)), vmax=np.max(np.abs(impulse)))
ax5.set_title("P2b Spatial Kernel", color='white', fontsize=14)
ax5.axis('off')
plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

ax6 = plt.subplot2grid((3, 4), (2, 2))
plot_intensity(ax6, intensity_p2b_no_pool, "P2b Raw Result (Log)")

ax7 = plt.subplot2grid((3, 4), (2, 3))
plot_intensity(ax7, intensity_p2b, "P2b Pooled (Log)")

plt.tight_layout()
plt.savefig("images/forward_pass_comparison.png", dpi=300, facecolor=fig.get_facecolor())
print("Saved images/forward_pass_comparison.png")
