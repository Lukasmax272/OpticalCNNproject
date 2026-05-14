import sys
sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math

from physics_sim import OpticalSimulator
from four_f_sim import MotionPoolingFourierMask

def get_field_at_z(E0, z, pixel_size, wavelength, device):
    """Angular spectrum propagation to distance z."""
    if z == 0:
        return E0
    N = E0.shape[-1]
    f = (torch.arange(N, dtype=torch.float32, device=device) - N // 2) / (N * pixel_size)
    FX, FY = torch.meshgrid(f, f, indexing="ij")
    
    inv_lam = 1.0 / wavelength
    kz_arg = inv_lam ** 2 - FX ** 2 - FY ** 2
    propagating = (kz_arg >= 0).to(torch.float32)
    
    kz = 2 * math.pi * torch.sqrt(torch.clamp(kz_arg, min=0.0))
    k = 2 * math.pi * inv_lam
    H = torch.exp(1j * (kz - k) * z) * propagating
    
    spectrum = torch.fft.fftshift(torch.fft.fft2(E0), dim=(-2, -1))
    spectrum = spectrum * H
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))

def visualize_phase1():
    print("Simulating Phase 1 propagation...")
    DEVICE = torch.device('cpu')
    GRID = 64
    PIXEL = 8e-6
    LAMBDA = 520e-9
    Z_TOTAL = 0.096
    
    # Load mask
    mask_1l = np.load("learned_phase_mask_1l_64p.npy")[0]
    mask_torch = torch.from_numpy(mask_1l).to(DEVICE)
    
    # Input: an MNIST digit (simplified for visualization: just a 3)
    # We'll create a synthetic '3' or load one
    from utils import load_mnist_generator
    gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
    E0, labels = next(gen)
    
    Z_STEPS = 40
    z1_range = np.linspace(0, Z_TOTAL/2, Z_STEPS)
    z2_range = np.linspace(0, Z_TOTAL/2, Z_STEPS)
    
    slices = []
    # Propagate to mask
    for z in z1_range:
        E = get_field_at_z(E0, z, PIXEL, LAMBDA, DEVICE)
        slices.append(E[0, GRID//2, :].numpy()) # x-z slice at y=middle
        
    # At mask
    E_mask = get_field_at_z(E0, Z_TOTAL/2, PIXEL, LAMBDA, DEVICE)
    E_modulated = E_mask * torch.exp(1j * mask_torch)
    
    # Propagate to detector
    for z in z2_range:
        E = get_field_at_z(E_modulated, z, PIXEL, LAMBDA, DEVICE)
        slices.append(E[0, GRID//2, :].numpy())
        
    vol = np.array(slices) # (Steps, GRID)
    intensity = np.abs(vol)**2
    
    return intensity, Z_TOTAL

def visualize_phase2b():
    print("Simulating Phase 2b (4f) propagation...")
    DEVICE = torch.device('cpu')
    GRID = 64
    PIXEL = 8e-6
    LAMBDA = 520e-9
    F = 0.05 # 5cm focal length
    
    # Load mask
    mask_p2b = np.load("learned_phase_mask_4f_motion_64.npy")
    H_p2b = torch.from_numpy(mask_p2b).to(DEVICE)
    
    from utils import load_mnist_generator
    gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
    E0, labels = next(gen)
    
    # 4f sequence: f -> Lens 1 -> f -> Mask -> f -> Lens 2 -> f -> Det
    def apply_lens(E, f):
        N = E.shape[-1]
        coords = (torch.arange(N, device=DEVICE) - N // 2) * PIXEL
        X, Y = torch.meshgrid(coords, coords, indexing="ij")
        k = 2 * math.pi / LAMBDA
        lens_phase = - (k / (2 * f)) * (X**2 + Y**2)
        return E * torch.exp(1j * lens_phase)

    Z_STEPS = 20
    all_slices = []
    
    # 1. Propagate f to Lens 1
    curr_E = E0
    for z in np.linspace(0, F, Z_STEPS):
        E = get_field_at_z(curr_E, z, PIXEL, LAMBDA, DEVICE)
        all_slices.append(E[0, GRID//2, :].numpy())
    
    # 2. Lens 1
    curr_E = get_field_at_z(curr_E, F, PIXEL, LAMBDA, DEVICE)
    curr_E = apply_lens(curr_E, F)
    
    # 3. Propagate f to Mask
    prev_E = curr_E
    for z in np.linspace(0, F, Z_STEPS):
        E = get_field_at_z(prev_E, z, PIXEL, LAMBDA, DEVICE)
        all_slices.append(E[0, GRID//2, :].numpy())
        
    # 4. Mask (Frequency plane)
    curr_E = get_field_at_z(prev_E, F, PIXEL, LAMBDA, DEVICE)
    # The mask is defined in the frequency plane. In a physical 4f, the field
    # in the Fourier plane IS the Fourier transform.
    # We apply the mask here. 
    curr_E = curr_E * torch.fft.ifftshift(H_p2b) # The H is DC-centred, but our field might not be?
    # Actually, in a physical 4f, the field at f distance from lens IS the FT.
    
    # 5. Propagate f to Lens 2
    prev_E = curr_E
    for z in np.linspace(0, F, Z_STEPS):
        E = get_field_at_z(prev_E, z, PIXEL, LAMBDA, DEVICE)
        all_slices.append(E[0, GRID//2, :].numpy())
        
    # 6. Lens 2
    curr_E = get_field_at_z(prev_E, F, PIXEL, LAMBDA, DEVICE)
    curr_E = apply_lens(curr_E, F)
    
    # 7. Propagate f to Detector
    prev_E = curr_E
    for z in np.linspace(0, F, Z_STEPS):
        E = get_field_at_z(prev_E, z, PIXEL, LAMBDA, DEVICE)
        all_slices.append(E[0, GRID//2, :].numpy())
        
    vol = np.array(all_slices)
    intensity = np.abs(vol)**2
    
    return intensity, 4*F

def create_animation(intensity, total_z, name, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#121212')
    
    # Side view: Intensity over Z and X
    # We want to show a moving "pulse" or just reveal the beam
    
    # Normalize
    intensity = intensity / intensity.max()
    
    im = ax.imshow(np.zeros((intensity.shape[1], intensity.shape[0])), 
                   extent=[0, total_z*100, -32*8, 32*8], # cm and um
                   cmap='magma', aspect='auto', origin='lower')
    
    ax.set_xlabel("Z Distance (cm)")
    ax.set_ylabel("X Position (µm)")
    ax.set_title(title)
    
    def update(frame):
        # Reveal up to frame
        data = np.zeros_like(intensity.T)
        data[:, :frame] = intensity[:frame].T
        im.set_data(data)
        return [im]
    
    ani = FuncAnimation(fig, update, frames=range(1, len(intensity)+1), blit=True)
    
    save_path = f"images/{name}.gif"
    os.makedirs("images", exist_ok=True)
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    i1, z1 = visualize_phase1()
    create_animation(i1, z1, "phase1_propagation", "Phase 1: Free-Space D2NN (1 Layer)")
    
    i2, z2 = visualize_phase2b()
    create_animation(i2, z2, "phase2b_propagation", "Phase 2b: 4f Correlator")
