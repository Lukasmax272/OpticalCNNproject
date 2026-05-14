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

def diagnose():
    DEVICE = torch.device('cpu')
    GRID = 64
    Z_TOTAL = 0.096
    
    # 1. Load masks
    mask_1l = np.load("learned_phase_mask_1l_64p.npy")[0]
    mask_p2b = np.load("learned_phase_mask_4f_motion_64.npy")
    
    # 2. Models
    spacings_1l = [Z_TOTAL/2, Z_TOTAL/2]
    sim_p1 = MultiLayerD2NN(num_layers=1, grid_size=GRID, layer_spacings=spacings_1l, device=DEVICE)
    sim_p1.phase_masks[0].data.copy_(torch.from_numpy(mask_1l))
    
    sim_p2b = MotionPoolingFourierMask(size=GRID, device=DEVICE)
    H_p2b = torch.from_numpy(mask_p2b).to(DEVICE)
    
    bin_masks = build_bin_masks(GRID, DEVICE)
    
    # 3. Find digit 0
    gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
    found_0 = None
    while found_0 is None:
        field, lbl = next(gen)
        label = lbl[0].item()
        if label == 0:
            with torch.no_grad():
                i1 = sim_p1(field)
                b1 = torch.einsum("bhw,khw->bk", i1, bin_masks)
                if b1.argmax(-1).item() == 0:
                    found_0 = (field, i1)
                    print("Found correctly classified Digit 0")

    # 4. Plot to see where the bright spot is
    field, intensity = found_0
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(field[0].abs().numpy(), cmap='gray')
    plt.title("Input Digit 0")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(intensity[0].numpy()), cmap='magma')
    plt.title("Output Intensity")
    
    # Draw bin centers
    layout = [
        (0, 1/6, 1/4), (1, 1/6, 2/4), (2, 1/6, 3/4),
        (3, 1/2, 1/5), (4, 1/2, 2/5), (5, 1/2, 3/5), (6, 1/2, 4/5),
        (7, 5/6, 1/4), (8, 5/6, 2/4), (9, 5/6, 3/4),
    ]
    for cls, ry, rx in layout:
        plt.text(rx*GRID, ry*GRID, str(cls), color='white', ha='center', va='center')
        
    plt.savefig("scratch/diagnose_alignment.png")
    print("Saved scratch/diagnose_alignment.png")

if __name__ == "__main__":
    diagnose()
