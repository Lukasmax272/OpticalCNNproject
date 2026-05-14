"""Quick sanity checks for Phase 1 validation (steps 1-3 of the roadmap).

Run: python diagnose.py
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

from physics_sim import OpticalSimulator
from optical_env import OpticalEnv
from utils import load_mnist_generator

GRID = 64
DEVICE = torch.device("cpu")

sim = OpticalSimulator(size=GRID, device=DEVICE)
gen = load_mnist_generator(batch_size=64, device=DEVICE, train=True)
env = OpticalEnv(sim, gen, device=DEVICE)

# ---- 1. Propagation timing
field, _ = next(gen)
mask = torch.zeros(GRID, GRID)
# warm-up
sim.propagate(field, mask)
t0 = time.perf_counter()
for _ in range(20):
    sim.propagate(field, mask)
ms = (time.perf_counter() - t0) / 20 * 1000
print(f"propagate latency: {ms:.2f} ms per batch of 64  (expect 2-10 ms)")

# ---- 2. Random-mask reward baseline
rng = np.random.default_rng(42)
rewards = []
for _ in range(20):
    action = rng.uniform(0, 2 * np.pi, (GRID, GRID)).astype(np.float32)
    _, r, _, _, _ = env.step(action)
    rewards.append(r)
print(f"Random-mask reward: mean={np.mean(rewards):.4f}  std={np.std(rewards):.4f}  (expect ~-2.30)")

# ---- 3. Bin intensity inspection
zero_action = np.zeros((GRID, GRID), dtype=np.float32)
obs, r, _, _, info = env.step(zero_action)
print(f"\nZero-mask step:")
print(f"  reward     = {r:.4f}")
print(f"  accuracy   = {info['accuracy']:.4f}")
print(f"  bins (avg) = {obs}")
print(f"  sum={obs.sum():.2f}  max={obs.max():.4f}  min={obs.min():.4f}")

# Also check what cross_entropy does with the raw bin values from a real step
phase = torch.zeros(GRID, GRID)
field2, labels2 = next(gen)
with torch.no_grad():
    intensity = sim.propagate(field2, phase)
    bins = env._extract_bins(intensity)

print(f"\nRaw bins shape: {bins.shape}")
print(f"  mean={bins.mean():.4f}  std={bins.std():.4f}  max={bins.max():.4f}  min={bins.min():.4f}")
ce = F.cross_entropy(bins, labels2)
print(f"  cross_entropy = {ce.item():.4f}  (expect ~2.30 for chance)")

# ---- 4. Advantage / gradient scale estimate
# With sigma=0.04, the policy gradient per dimension is scaled by 1/sigma^2
sigma = 0.04
print(f"\nGradient scaling analysis:")
print(f"  sigma^2 = {sigma**2:.6f}")
print(f"  1/sigma^2 (gradient amplifier) = {1/sigma**2:.1f}")
print(f"  With lr=0.003: param update per step ~ {0.003 / sigma**2:.3f}")
print(f"  KL per dim (assuming update=0.003/sigma^2): {(0.003/sigma**2)**2 / (2*sigma**2):.2f}")
print(f"  Total KL (4096 dims): {4096 * (0.003/sigma**2)**2 / (2*sigma**2):.2e}  (observed ~3.5e7)")
