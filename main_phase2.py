"""Phase 2 — direct backprop training with 4f correlator + peak-sensitive detector.

The phase mask is an nn.Parameter of shape (GRID, GRID) trained via Adam.
Architectural differences from Phase 1:

  1. Propagation: 4f correlator (FFT → SLM in Fourier plane → IFFT).
     Gives exact translation equivariance: shifting the input shifts
     the output intensity pattern identically.

  2. Detection: amax over enlarged bin regions.  Each bin is the same
     layout as Phase 1, expanded by `margin = max_shift` pixels so the
     peak stays inside the bin across the full shift range.

  3. Initialization: binary phase (0 or π) for maximum modulation contrast.

  4. LR scaling: lr is scaled by min(1, 64/grid) for larger grids, since
     the Fourier-plane mask has a different curvature landscape.

  5. Shift curriculum: shifts ramp from 0 to max_shift over the first
     half of training, letting the mask learn digit features before
     learning shift robustness.

The entire pipeline is optical — no digital layers.

Run:
    python main_phase2.py [--grid 256] [--max_shift 8] [--iters 10000]
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F

from four_f_sim import FourFSimulator
from utils import load_mnist_generator
from main import build_bin_masks


DEVICE = torch.device("cpu")


def curriculum_shift(it: int, total_iters: int, max_shift: int) -> int:
    """Shift curriculum: no shift for first 10%, ramp to max_shift by 50%."""
    warmup = total_iters // 10
    if it < warmup:
        return 0
    ramp = min(1.0, (it - warmup) / (total_iters * 0.4))
    return max(1, round(ramp * max_shift))


def apply_random_shifts(field: torch.Tensor, current_shift: int) -> torch.Tensor:
    """Apply independent random (dy, dx) shift to each image in the batch."""
    for i in range(field.shape[0]):
        dy = torch.randint(-current_shift, current_shift + 1, (1,)).item()
        dx = torch.randint(-current_shift, current_shift + 1, (1,)).item()
        field[i] = torch.roll(field[i], shifts=(dy, dx), dims=(-2, -1))
    return field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64,
                        help="Grid size (default: 64, try 128 or 256)")
    parser.add_argument("--max_shift", type=int, default=0,
                        help="Max random translation in pixels")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Base LR (auto-scaled by 64/grid)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (default: 64 for grid<=64, 128 otherwise)")
    args = parser.parse_args()

    grid = args.grid
    batch = args.batch if args.batch > 0 else (64 if grid <= 64 else 128)
    effective_lr = args.lr * min(1.0, 64.0 / grid)

    sim = FourFSimulator(size=grid, device=DEVICE)
    # Generator yields unshifted data; curriculum applies shifts in the loop.
    gen = load_mnist_generator(batch_size=batch, device=DEVICE, train=True,
                               grid_size=grid)

    # Enlarged bins absorb the full shift range.
    bin_masks = build_bin_masks(grid, DEVICE, margin=args.max_shift)

    # Binary phase init: 0 or π for maximum modulation contrast.
    phase_mask = torch.nn.Parameter(
        torch.randint(0, 2, (grid, grid), device=DEVICE).float() * math.pi
    )
    optimizer = torch.optim.Adam([phase_mask], lr=effective_lr)

    accs: list[float] = []
    losses: list[float] = []
    LOG_EVERY = 100

    print(f"Grid: {grid}x{grid}  LR: {effective_lr:.4f}  Batch: {batch}  "
          f"Shift: 0->{args.max_shift} (curriculum)  Iters: {args.iters}")

    for it in range(args.iters):
        field, labels = next(gen)

        # Shift curriculum: ramp shift magnitude over first half of training.
        if args.max_shift > 0:
            cs = curriculum_shift(it, args.iters, args.max_shift)
            if cs > 0:
                field = apply_random_shifts(field, cs)

        intensity = sim.propagate(field, phase_mask)

        # Peak-sensitive detector: amax within each enlarged bin region.
        # If training becomes noisy at large grids, replace amax with
        # soft-max pooling: softmax(intensity/τ) weighted sum, τ~0.1.
        masked_intensity = intensity.unsqueeze(1) * bin_masks.unsqueeze(0)
        bins = masked_intensity.amax(dim=(-2, -1))

        normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        loss = F.cross_entropy(normalized, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (bins.argmax(-1) == labels).float().mean().item()
        accs.append(acc)
        losses.append(loss.item())

        if (it + 1) % LOG_EVERY == 0:
            recent_acc = sum(accs[-LOG_EVERY:]) / LOG_EVERY
            recent_loss = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            print(
                f"iter {it + 1:5d}  loss={recent_loss:.4f}  "
                f"acc(last {LOG_EVERY})={recent_acc:.3f}"
            )

    mask_path = f"learned_phase_mask_4f_{grid}.npy"
    np.save(mask_path, phase_mask.detach().cpu().numpy())
    print(f"\nSaved {mask_path}  shape={phase_mask.shape}")
    final_acc = sum(accs[-LOG_EVERY:]) / LOG_EVERY
    print(f"Final accuracy (last {LOG_EVERY} iters): {final_acc:.4f}")


if __name__ == "__main__":
    main()
