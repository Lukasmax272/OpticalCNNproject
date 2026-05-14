import sys
sys.path.append(".")
import subprocess
import torch
import numpy as np
from eval_shifted import evaluate_at_shift
from physics_sim import MultiLayerD2NN
from main import build_bin_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = 128
ITERS = 10000

for layers in [1, 3]:
    print(f"\n=============================================")
    print(f"Training {layers}-layer D2NN with +/- 8 shifts (128x128)")
    print(f"=============================================")
    # Train
    subprocess.run([
        ".venv\\Scripts\\python.exe", "main.py", 
        "--num-layers", str(layers), 
        "--max-shift", "8",
        "--grid", str(GRID),
        "--iters", str(ITERS)
    ], check=True)
    
    # Load model and evaluate
    z_total = 0.096
    spacings = [z_total / (layers + 1)] * (layers + 1)
    sim = MultiLayerD2NN(num_layers=layers, grid_size=GRID, layer_spacings=spacings, device=DEVICE)
    # The saved mask depends on relu. Since we didn't pass relu, it's learned_phase_mask.npy
    masks = np.load("learned_phase_mask.npy")
    for i in range(layers):
        sim.phase_masks[i].data.copy_(torch.from_numpy(masks[i]).to(DEVICE))
        
    bin_masks = build_bin_masks(GRID, DEVICE, margin=0)
    
    print(f"\n--- Evaluation for {layers}-layer D2NN ---")
    for s in [0, 4, -4, 8, -8]:
        acc = evaluate_at_shift(sim, None, bin_masks, s, s, use_amax=False, grid_size=GRID)
        print(f"  shift={s:+3d}  acc={acc:.4f}")
    
    print("-" * 30)
