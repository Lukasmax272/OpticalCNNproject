"""Compare random-phase-init vs zero-phase-init training on the 3-row bin layout.

Runs 5000 iterations for each initialization, then reports:
  - Final training accuracy (last 100 iters)
  - Final bin power coverage (fraction of field power captured by all 10 bins)
  - Whether the anti-bin routing effect is still present

Both runs use identical hyperparameters. The only variable is the starting
value of phase_mask.
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from physics_sim import OpticalSimulator
from utils import load_mnist_generator
from main import build_bin_masks

GRID = 64
DATA_BATCH = 64
TOTAL_ITERATIONS = 5000
LEARNING_RATE = 0.01
LOG_EVERY = 500   # less verbose than main.py for side-by-side readability
DEVICE = torch.device("cpu")


def train(init: str, coverage_loss: bool = False, coverage_lambda: float = 1.0) -> dict:
    """Run one full training cycle.  init is 'random' or 'zero'."""
    sim = OpticalSimulator(size=GRID, device=DEVICE)
    # Fresh generator each run so both see the same data order.
    gen = load_mnist_generator(batch_size=DATA_BATCH, device=DEVICE, train=True)
    bin_masks = build_bin_masks(GRID, DEVICE)

    if init == "random":
        phase_mask = torch.nn.Parameter(torch.rand(GRID, GRID) * 2 * math.pi)
    else:
        phase_mask = torch.nn.Parameter(torch.zeros(GRID, GRID))

    optimizer = torch.optim.Adam([phase_mask], lr=LEARNING_RATE)

    accs, losses = [], []
    t0 = time.perf_counter()

    for it in range(TOTAL_ITERATIONS):
        field, labels = next(gen)

        intensity = sim.propagate(field, phase_mask)
        bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
        normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        loss = F.cross_entropy(normalized, labels)

        if coverage_loss:
            total_power = intensity.sum(dim=(-2, -1))
            correct_bin = bins[torch.arange(len(labels)), labels]
            target_coverage = correct_bin / total_power.clamp(min=1e-8)
            loss = loss - coverage_lambda * target_coverage.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (bins.argmax(-1) == labels).float().mean().item()
        accs.append(acc)
        losses.append(loss.item())

        if (it + 1) % LOG_EVERY == 0:
            ra = sum(accs[-LOG_EVERY:]) / LOG_EVERY
            rl = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            cov_tag = f"lam={coverage_lambda}" if coverage_loss else "no-cov"
            print(f"  [{init:6s} {cov_tag}] iter {it+1:5d}  loss={rl:.4f}  acc={ra:.3f}")

    elapsed = time.perf_counter() - t0

    # Measure final bin coverage on a held-out batch.
    gen_test = load_mnist_generator(batch_size=256, device=DEVICE, train=False)
    field_t, _ = next(gen_test)
    with torch.no_grad():
        intensity_t = sim.propagate(field_t, phase_mask)
        bins_t = torch.einsum("bhw,khw->bk", intensity_t, bin_masks)
        total_power = intensity_t.sum(dim=(-2, -1))
        bin_power = bins_t.sum(-1)
        coverage = (bin_power / total_power).mean().item()

    final_acc = sum(accs[-100:]) / 100

    np.save(f"learned_phase_mask_{init}.npy", phase_mask.detach().cpu().numpy())

    return {
        "init": init,
        "final_acc": final_acc,
        "coverage": coverage,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    import sys
    # Usage: python compare_inits.py [coverage_lambda]
    # Default: runs random-no-cov vs random-with-cov (lam=1.0 or supplied value)
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

    print("=== Random init, no coverage loss ===")
    r1 = train("random", coverage_loss=False)
    print()
    print(f"=== Random init, coverage loss lam={lam} ===")
    r2 = train("random", coverage_loss=True, coverage_lambda=lam)

    print("\n" + "=" * 65)
    print(f"{'':28s}  {'no-coverage':>14s}  {f'coverage lam={lam}':>14s}")
    print(f"{'Final train acc':28s}  {r1['final_acc']:>14.4f}  {r2['final_acc']:>14.4f}")
    print(f"{'Bin coverage':28s}  {r1['coverage']:>14.4f}  {r2['coverage']:>14.4f}")
    print(f"{'Wall time (s)':28s}  {r1['elapsed_s']:>14.1f}  {r2['elapsed_s']:>14.1f}")
    print("=" * 65)
    print("Saved learned_phase_mask_random.npy  (no-coverage run)")
