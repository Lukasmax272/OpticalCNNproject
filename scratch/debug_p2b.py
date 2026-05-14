import sys
sys.path.append(".")
import numpy as np
import torch
from main import build_bin_masks
from four_f_sim import MotionPoolingFourierMask

def debug_p2b():
    DEVICE = torch.device('cpu')
    GRID = 64
    sim = MotionPoolingFourierMask(size=GRID, device=DEVICE)
    mask = np.load("learned_phase_mask_4f_motion_64.npy")
    H = torch.from_numpy(mask).to(DEVICE)
    
    bin_masks = build_bin_masks(GRID, DEVICE)
    from utils import load_mnist_generator
    gen = load_mnist_generator(batch_size=1, device=DEVICE, train=False, grid_size=GRID)
    
    found = 0
    while found < 5:
        field, lbl = next(gen)
        label = lbl[0].item()
        
        with torch.no_grad():
            intensity = sim.propagate(field, H)
            bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)[0]
            pred = bins.argmax().item()
            print(f"Digit {label} -> Predicted {pred} (Bins: {bins.numpy().round(1)})")
        found += 1

if __name__ == "__main__":
    debug_p2b()
