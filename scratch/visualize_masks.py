import os
import numpy as np
import matplotlib.pyplot as plt
import torch

os.makedirs('images', exist_ok=True)

# 1. Visualize Phase 2b Mask
print("Loading Phase 2b mask...")
mask_p2b = np.load("learned_phase_mask_4f_motion_64.npy")
amplitude = np.abs(mask_p2b)
phase = np.angle(mask_p2b)

# Spatial impulse response
# H is DC-centered in the Fourier domain.
H_tensor = torch.from_numpy(mask_p2b)
impulse_response = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(H_tensor))).real.numpy()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(amplitude, cmap='magma')
axs[0].set_title("Fourier Amplitude |H|")
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axs[1].set_title("Fourier Phase angle(H)")
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(impulse_response, cmap='bwr', vmin=-np.max(np.abs(impulse_response)), vmax=np.max(np.abs(impulse_response)))
axs[2].set_title("Spatial Impulse Response")
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.suptitle("Phase 2b: Hermitian-Symmetric 4f Mask (Translation Equivariant)")
plt.tight_layout()
plt.savefig("images/phase2b_mask.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Visualize Phase 1 1-layer Mask
print("Loading Phase 1 (1-layer) mask...")
mask_1l = np.load("learned_phase_mask_1l_64p.npy")[0] # Shape (1, 64, 64)
mask_1l_wrapped = np.mod(mask_1l, 2 * np.pi)

plt.figure(figsize=(6, 5))
im = plt.imshow(mask_1l_wrapped, cmap='twilight', vmin=0, vmax=2*np.pi)
plt.title("Phase 1: 1-Layer D²NN Spatial Phase Mask")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("images/phase1_1layer_mask.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Visualize Phase 1 3-layer Mask
print("Loading Phase 1 (3-layer) mask...")
mask_3l = np.load("learned_phase_mask_3l_64p.npy") # Shape (3, 64, 64)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    mask_3l_wrapped = np.mod(mask_3l[i], 2 * np.pi)
    im = axs[i].imshow(mask_3l_wrapped, cmap='twilight', vmin=0, vmax=2*np.pi)
    axs[i].set_title(f"Layer {i+1}")
    plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

plt.suptitle("Phase 1b: 3-Layer D²NN Spatial Phase Masks")
plt.tight_layout()
plt.savefig("images/phase1_3layer_masks.png", dpi=300, bbox_inches='tight')
plt.close()

print("Images saved successfully to the 'images/' directory.")
