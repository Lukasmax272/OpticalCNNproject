import sys
sys.path.append(".")
import torch
import numpy as np
import os
import math

from physics_sim import MultiLayerD2NN
from four_f_sim import MotionPoolingFourierMask
from utils import load_mnist_generator
from main import build_bin_masks

def find_global_orientation():
    DEVICE = torch.device('cpu')
    GRID = 64
    
    mask_1l = np.load("learned_phase_mask_1l_64p.npy")[0]
    mask_p2b = np.load("learned_phase_mask_4f_motion_64.npy")
    
    sim_p2b = MotionPoolingFourierMask(size=GRID, device=DEVICE)
    H_p2b = torch.from_numpy(mask_p2b).to(DEVICE)
    
    sim_p1 = MultiLayerD2NN(num_layers=1, grid_size=GRID, layer_spacings=[0.048, 0.048], device=DEVICE)
    sim_p1.phase_masks[0].data.copy_(torch.from_numpy(mask_1l))
    
    bin_masks_np = build_bin_masks(GRID, DEVICE).numpy()
    
    orientations = [
        lambda x: x, lambda x: x.T, lambda x: np.flipud(x), lambda x: np.fliplr(x),
        lambda x: np.flipud(x.T), lambda x: np.fliplr(x.T), lambda x: np.flipud(np.fliplr(x)),
        lambda x: np.flipud(np.fliplr(x.T))
    ]
    orient_names = ["Identity", "Transpose", "FlipUD", "FlipLR", "FlipUD+T", "FlipLR+T", "180-Rot", "180-Rot+T"]
    
    counts_p1 = np.zeros(8)
    counts_p2b = np.zeros(8)
    
    gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
    print("Testing 100 digits...")
    for _ in range(100):
        field, lbl = next(gen)
        label = lbl[0].item()
        
        with torch.no_grad():
            i1 = sim_p1(field)[0].numpy()
            i2b = sim_p2b.propagate(field, H_p2b)[0].numpy()
            
            for i, orient in enumerate(orientations):
                if np.argmax([np.sum(orient(i1) * bin_masks_np[k]) for k in range(10)]) == label:
                    counts_p1[i] += 1
                if np.argmax([np.sum(orient(i2b) * bin_masks_np[k]) for k in range(10)]) == label:
                    counts_p2b[i] += 1
                    
    print("\nPhase 1 Results:")
    for i, name in enumerate(orient_names):
        print(f"  {name}: {int(counts_p1[i])}/100")
        
    print("\nPhase 2b Results:")
    for i, name in enumerate(orient_names):
        print(f"  {name}: {int(counts_p2b[i])}/100")

if __name__ == "__main__":
    find_global_orientation()
