import sys
sys.path.append(".")
import torch
import numpy as np
import os
import math
import json
from scipy.ndimage import gaussian_filter

from physics_sim import MultiLayerD2NN
from four_f_sim import MotionPoolingFourierMask
from four_f_hybrid import ODCNNHybrid, build_detector_masks_200
from utils import load_mnist_generator

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

def generate_data():
    DEVICE = torch.device('cpu')
    GRID_WEB = 100 # We'll scale everything to 100x100 for browser performance
    NUM_DIGITS = 10
    Z_STEPS = 50
    
    os.makedirs("web_data", exist_ok=True)
    
    # 1. Load Phase 1 Mask (64x64)
    mask_1l = np.load("learned_phase_mask_1l_64p.npy")[0]
    
    # 2. Load Phase 2b Mask (64x64)
    mask_p2b = np.load("learned_phase_mask_4f_motion_64.npy")
    
    # 3. Load Phase 2c Model (200x200)
    ckpt = torch.load("results/odcnn_hybrid.pt", map_location=DEVICE)
    sim_p2c = ODCNNHybrid(
        grid_size=ckpt["grid_size"],
        num_kernels=ckpt["num_kernels"],
        kernel_size=ckpt["kernel_size"],
        d2nn_layers=ckpt["d2nn_layers"],
        d2nn_z_total=ckpt["d2nn_z_total"],
        pixel_size=ckpt["pixel_size"],
        motion_pool_sigma=ckpt.get("motion_pool_sigma", 0.0),
    ).to(DEVICE)
    with torch.no_grad():
        sim_p2c.tiled_mask.spatial_kernels.copy_(ckpt["spatial_kernels"])
        for i, al in enumerate(ckpt["amplitude_logits"]):
            sim_p2c.d2nn.amplitude_logits[i].copy_(al)
        for i, pm in enumerate(ckpt["phase_masks"]):
            sim_p2c.d2nn.phase_masks[i].copy_(pm)
    sim_p2c.eval()
    
    # Detector masks for Phase 2c
    bin_masks_p2c = build_detector_masks_200(200, DEVICE)
    
    # 4. Load Phase 1/2b models for verification
    Z_TOTAL_P1 = 0.096
    sim_p1 = MultiLayerD2NN(num_layers=1, grid_size=64, layer_spacings=[Z_TOTAL_P1/2]*2, device=DEVICE)
    sim_p1.phase_masks[0].data.copy_(torch.from_numpy(mask_1l))
    
    sim_p2b = MotionPoolingFourierMask(size=64, device=DEVICE)
    H_p2b = torch.from_numpy(mask_p2b).to(DEVICE)
    
    # 5. Find digits correctly classified by ALL models
    # Note: P1/P2b use 64x64 phase encoding. P2c uses 200x200 amplitude encoding.
    gen_64 = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=64, amplitude_encode=False)
    gen_200 = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=200, amplitude_encode=True, upsample=False)
    
    # We'll search using a common seed or just loop until we find samples for each class
    digits_field_64 = [None] * 10
    digits_field_200 = [None] * 10
    found_count = 0
    
    print("Searching for digits correctly classified by all three models...")
    # This might take a while if we want exact matches, so let's just find "good" samples per class
    # even if they aren't the exact same digit image (though it would be nicer).
    # To keep it simple, we'll find samples for each model independently but ensure we have one of each class.
    
    p1_ok = [False] * 10
    p2b_ok = [False] * 10
    p2c_ok = [False] * 10
    
    while not all(p1_ok) or not all(p2b_ok) or not all(p2c_ok):
        # Sample from 64
        f64, l64 = next(gen_64)
        lbl = l64[0].item()
        if not p1_ok[lbl] or not p2b_ok[lbl]:
            with torch.no_grad():
                # P1 check
                i1 = sim_p1(f64)
                from main import build_bin_masks
                bm64 = build_bin_masks(64, DEVICE)
                b1 = torch.einsum("bhw,khw->bk", i1, bm64)
                if b1.argmax().item() == lbl and not p1_ok[lbl]:
                    digits_field_64[lbl] = f64
                    p1_ok[lbl] = True
                
                # P2b check
                i2b = sim_p2b.propagate(f64, H_p2b)
                if torch.einsum("bhw,khw->bk", i2b, bm64).argmax().item() == lbl and not p2b_ok[lbl]:
                    # We'll use the same 64x64 field if possible
                    p2b_ok[lbl] = True
        
        # Sample from 200
        f200, l200 = next(gen_200)
        lbl = l200[0].item()
        if not p2c_ok[lbl]:
            with torch.no_grad():
                i2c = sim_p2c(f200)
                b2c = torch.einsum("bhw,khw->bk", i2c, bin_masks_p2c)
                if b2c.argmax().item() == lbl:
                    digits_field_200[lbl] = f200
                    p2c_ok[lbl] = True
        
        print(f"\r  Progress: P1:{sum(p1_ok)}/10, P2b:{sum(p2b_ok)}/10, P2c:{sum(p2c_ok)}/10", end="")
    print("\nDone searching.")

    # Storage
    p1_data = np.zeros((NUM_DIGITS, Z_STEPS, GRID_WEB, GRID_WEB), dtype=np.uint8)
    p2b_data = np.zeros((NUM_DIGITS, Z_STEPS, GRID_WEB, GRID_WEB), dtype=np.uint8)
    p2c_data = np.zeros((NUM_DIGITS, Z_STEPS, GRID_WEB, GRID_WEB), dtype=np.uint8)

    def normalize_and_scale(intensity, target_size=100):
        # intensity is (H, W)
        # Use log1p to boost signal
        log_i = np.log1p(intensity.numpy() * 100)
        norm = (255 * (log_i / (log_i.max() + 1e-8))).astype(np.uint8)
        # Resize to target_size
        img = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        return resized.squeeze().numpy().astype(np.uint8)

    # 6. Generate Phase 1 Data
    print("Generating Phase 1 data...")
    for d in range(10):
        field = digits_field_64[d]
        mask_torch = torch.from_numpy(mask_1l).to(DEVICE)
        with torch.no_grad():
            for i in range(25):
                E = get_field_at_z(field, (i/25)*(Z_TOTAL_P1/2), 8e-6, 520e-9, DEVICE)
                p1_data[d, i] = normalize_and_scale(E.abs().square()[0])
            E_mask = get_field_at_z(field, Z_TOTAL_P1/2, 8e-6, 520e-9, DEVICE)
            E_mod = E_mask * torch.exp(1j * mask_torch)
            for i in range(25, 50):
                E = get_field_at_z(E_mod, ((i-25)/24)*(Z_TOTAL_P1/2), 8e-6, 520e-9, DEVICE)
                p1_data[d, i] = normalize_and_scale(E.abs().square()[0])

    # 7. Generate Phase 2b Data
    print("Generating Phase 2b data...")
    for d in range(10):
        field = digits_field_64[d]
        with torch.no_grad():
            spec = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
            modulated = spec * H_p2b
            out_raw = torch.fft.ifft2(torch.fft.ifftshift(modulated, dim=(-2, -1)))
            i_raw = out_raw.abs().square()[0]
            # Stage 1: Propagation (0-10)
            for i in range(11):
                E = get_field_at_z(field, (i/10)*0.01, 8e-6, 520e-9, DEVICE)
                p2b_data[d, i] = normalize_and_scale(E.abs().square()[0])
            # Stage 2: FFT Transition (11-25)
            for i in range(11, 26):
                p = (i-11)/14
                it = (1-p)*field.abs().square()[0] + p*spec.abs().square()[0]
                p2b_data[d, i] = normalize_and_scale(it)
            # Stage 3: At Mask (26-35)
            for i in range(26, 36):
                p2b_data[d, i] = normalize_and_scale(modulated.abs().square()[0])
            # Stage 4: IFFT Transition (36-49)
            for i in range(36, 50):
                p = (i-36)/13
                it = (1-p)*modulated.abs().square()[0] + p*i_raw
                p2b_data[d, i] = normalize_and_scale(it)

    # 8. Generate Phase 2c Data
    print("Generating Phase 2c data...")
    # Tiled 4f -> 5-layer D2NN
    # We have 50 steps.
    # 0-10: 4f stage (Input -> Spectrum -> Filtered -> Conv)
    # 11-49: D2NN stage (5 layers) -> approx 8 steps per layer
    for d in range(10):
        field = digits_field_200[d]
        with torch.no_grad():
            # Stage 1: 4f
            spectrum = torch.fft.fft2(field)
            H = sim_p2c.tiled_mask.get_fourier_mask()
            modulated = spectrum * H
            E_conv = torch.fft.ifft2(modulated)
            
            # Sub-steps for 4f
            for i in range(4): # 0-3: Input to Spectrum
                p = i/3
                it = (1-p)*field.abs().square()[0] + p*torch.fft.fftshift(spectrum, dim=(-2, -1)).abs().square()[0]
                p2c_data[d, i] = normalize_and_scale(it)
            for i in range(4, 7): # 4-6: Spectrum at Mask
                p = (i-4)/2
                it = (1-p)*torch.fft.fftshift(spectrum, dim=(-2, -1)).abs().square()[0] + p*torch.fft.fftshift(modulated, dim=(-2, -1)).abs().square()[0]
                p2c_data[d, i] = normalize_and_scale(it)
            for i in range(7, 11): # 7-10: Mask to Conv
                p = (i-7)/3
                it = (1-p)*torch.fft.fftshift(modulated, dim=(-2, -1)).abs().square()[0] + p*E_conv.abs().square()[0]
                p2c_data[d, i] = normalize_and_scale(it)
            
            # Stage 2: D2NN (11-49)
            E = E_conv
            for layer_idx in range(5):
                start_step = 11 + layer_idx * 8
                end_step = min(50, start_step + 8)
                
                # Propagate to layer
                E_pre = sim_p2c.d2nn._propagate(E)
                # Modulation
                amp = torch.sigmoid(sim_p2c.d2nn.amplitude_logits[layer_idx])
                phi = sim_p2c.d2nn.phase_masks[layer_idx]
                t = amp * torch.exp(1j * phi)
                E_post = E_pre * t
                
                for i in range(start_step, end_step):
                    p = (i - start_step) / (end_step - start_step - 1) if (end_step - start_step) > 1 else 1.0
                    it = (1-p)*E_pre.abs().square()[0] + p*E_post.abs().square()[0]
                    p2c_data[d, i] = normalize_and_scale(it)
                
                E = E_post
            
            # Final propagation to detector
            E_final = sim_p2c.d2nn._propagate(E)
            p2c_data[d, 49] = normalize_and_scale(E_final.abs().square()[0])

    import base64
    import matplotlib.cm as cm
    magma_lut = (cm.magma(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8).tolist()
    p1_b64 = base64.b64encode(p1_data.tobytes()).decode('utf-8')
    p2b_b64 = base64.b64encode(p2b_data.tobytes()).decode('utf-8')
    p2c_b64 = base64.b64encode(p2c_data.tobytes()).decode('utf-8')
    
    meta = {
        "num_digits": NUM_DIGITS, "z_steps": Z_STEPS, "grid_size": GRID_WEB,
        "labels": list(range(10)), "p1_z_total": Z_TOTAL_P1, "p2b_z_total": 0.2, "p2c_z_total": 0.3
    }
    with open("browser/data.js", "w") as f:
        f.write(f"const META = {json.dumps(meta)};\n")
        f.write(f"const MAGMA_LUT = {json.dumps(magma_lut)};\n")
        f.write(f"const P1_DATA_B64 = '{p1_b64}';\n")
        f.write(f"const P2B_DATA_B64 = '{p2b_b64}';\n")
        f.write(f"const P2C_DATA_B64 = '{p2c_b64}';\n")
    print("\nDone. Data saved in browser/data.js")

if __name__ == "__main__":
    generate_data()
