import subprocess

print("==================================================")
print("Running Phase 1 (1-Layer D2NN) | 128x128 | 10k iters")
print("==================================================")
subprocess.run([
    ".venv\\Scripts\\python.exe", "main.py", 
    "--num-layers", "1", 
    "--max-shift", "8",
    "--grid", "128",
    "--iters", "10000"
], check=True)

print("\n==================================================")
print("Running Phase 2b (4f Motion Pooling) | 128x128 | 10k iters")
print("==================================================")
subprocess.run([
    ".venv\\Scripts\\python.exe", "main_phase2_motion_pool.py", 
    "--grid", "128",
    "--shift-range", "8",
    "--iters", "10000",
    "--motion-pool-sigma", "6.0"  # scale sigma for 128 grid (3.0 for 64 -> 6.0 for 128)
], check=True)

print("\nEvaluating Phase 2b...")
subprocess.run([
    ".venv\\Scripts\\python.exe", "eval_shifted.py",
    "learned_phase_mask_4f_motion_128.npy",
    "--arch", "phase2b",
    "--grid", "128",
    "--sigma", "6.0",
    "--shifts", "0", "4", "-4", "8", "-8"
], check=True)
