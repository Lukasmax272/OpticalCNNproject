import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

# Train Phase 1 (1-Layer D2NN)
print("Training 1-Layer D2NN (Phase 1)...")
subprocess.run([
    ".venv\\Scripts\\python.exe", "main.py", 
    "--num-layers", "1", 
    "--max-shift", "5",
    "--grid", "64",
    "--iters", "5000"
], check=True)

# Train Phase 1b (3-Layer D2NN)
print("\nTraining 3-Layer D2NN (Phase 1b)...")
subprocess.run([
    ".venv\\Scripts\\python.exe", "main.py", 
    "--num-layers", "3", 
    "--max-shift", "5",
    "--grid", "64",
    "--iters", "5000"
], check=True)

# Train Phase 2b (4f Motion Pooling)
print("\nTraining 1-Layer 4f Motion Pooling (Phase 2b)...")
subprocess.run([
    ".venv\\Scripts\\python.exe", "main_phase2_motion_pool.py", 
    "--grid", "64",
    "--shift-range", "5",
    "--iters", "5000",
    "--motion-pool-sigma", "3.0"
], check=True)

# Plot
print("\nGenerating plot...")
hist_1l = np.load("history_1l_64p.npy", allow_pickle=True).item()
hist_3l = np.load("history_3l_64p.npy", allow_pickle=True).item()
hist_p2 = np.load("history_phase2b_64p.npy", allow_pickle=True).item()

def smooth(arr, window=100):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 6))

plt.plot(smooth(hist_1l["accs"]), label='1-Layer D²NN (Phase 1)')
plt.plot(smooth(hist_3l["accs"]), label='3-Layer D²NN (Phase 1b)')
plt.plot(smooth(hist_p2["accs"]), label='1-Layer 4f + Motion Pooling (Phase 2b)')

plt.title('Training Accuracy vs Iterations (64x64, fully randomized ±5px shifts)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (smoothed)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

os.makedirs('artifacts', exist_ok=True)
plt.savefig('artifacts/training_comparison_64p.png', dpi=300, bbox_inches='tight')
print("Saved artifacts/training_comparison_64p.png")
