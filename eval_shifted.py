"""Evaluate any saved phase mask on shifted test data.

Usage:
    python eval_shifted.py <mask.npy> [--arch phase1|phase2] [--grid 128]
                                      [--shifts 0 4 -4 8 -8] [--margin 8]

Prints accuracy at each shift value and a summary table.
"""

import argparse
import numpy as np
import torch

from physics_sim import OpticalSimulator, MultiLayerD2NN
from four_f_sim import FourFSimulator, MotionPoolingFourierMask
from utils import load_mnist_generator
from main import build_bin_masks


DEVICE = torch.device("cpu")


def evaluate_at_shift(sim, phase_mask, bin_masks, shift_x, shift_y,
                      use_amax=False, grid_size=64,
                      n_batches=40, batch_size=256):
    """Evaluate accuracy at a fixed (shift_x, shift_y) translation."""
    gen = load_mnist_generator(batch_size=batch_size, device=DEVICE,
                               train=False, grid_size=grid_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            field, labels = next(gen)
            if shift_x != 0 or shift_y != 0:
                field = torch.roll(field, shifts=(shift_y, shift_x), dims=(-2, -1))
            if hasattr(sim, "propagate"):
                intensity = sim.propagate(field, phase_mask)
            else:
                intensity = sim(field)
            if use_amax:
                masked = intensity.unsqueeze(1) * bin_masks.unsqueeze(0)
                bins = masked.amax(dim=(-2, -1))
            else:
                bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            correct += (bins.argmax(-1) == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate mask on shifted test data.")
    parser.add_argument("mask", help="Path to .npy phase mask file")
    parser.add_argument("--arch", choices=["phase1", "phase2", "phase2b"], default="phase1",
                        help="Architecture: phase1, phase2, phase2b")
    parser.add_argument("--grid", type=int, default=0,
                        help="Grid size (default: inferred from mask shape)")
    parser.add_argument("--margin", type=int, default=8,
                        help="Bin margin for Phase 2 (should match training max_shift)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Sigma for Phase 2b motion pooling")
    parser.add_argument("--shifts", nargs="+", type=int, default=[0, -4, 4, -8, 8],
                        help="Shift values to test (applied to both x and y)")
    args = parser.parse_args()

    phase_mask = torch.from_numpy(np.load(args.mask)).to(DEVICE)
    grid = args.grid if args.grid > 0 else phase_mask.shape[0]

    if args.arch == "phase2":
        sim = FourFSimulator(size=grid, device=DEVICE)
        bin_masks = build_bin_masks(grid, DEVICE, margin=args.margin)
        use_amax = True
    elif args.arch == "phase2b":
        sim = MotionPoolingFourierMask(size=grid, sigma=args.sigma, device=DEVICE)
        bin_masks = build_bin_masks(grid, DEVICE, margin=0)
        use_amax = False
    else:
        # Phase 1 is trained using MultiLayerD2NN
        num_layers = phase_mask.shape[0] if phase_mask.ndim == 3 else 1
        z_total = 0.096
        spacings = [z_total / (num_layers + 1)] * (num_layers + 1)
        sim = MultiLayerD2NN(num_layers=num_layers, grid_size=grid, layer_spacings=spacings, device=DEVICE)
        if phase_mask.ndim == 3:
            for i in range(num_layers):
                sim.phase_masks[i].data.copy_(phase_mask[i])
        else:
            sim.phase_masks[0].data.copy_(phase_mask)
        bin_masks = build_bin_masks(grid, DEVICE, margin=0)
        use_amax = False

    print(f"Architecture: {args.arch}")
    print(f"Phase mask:   {args.mask}  shape={phase_mask.shape}")
    print(f"Grid: {grid}  Margin: {args.margin if use_amax else 'N/A'}")
    print(f"{'shift':>8s}  {'accuracy':>10s}")
    print("-" * 22)

    results = {}
    for s in args.shifts:
        acc = evaluate_at_shift(sim, phase_mask, bin_masks, s, s,
                                use_amax=use_amax, grid_size=grid)
        results[s] = acc
        print(f"{s:>8d}  {acc:>10.4f}")

    # Also test random shifts.
    gen = load_mnist_generator(batch_size=256, device=DEVICE, train=False,
                               grid_size=grid)
    correct = 0
    total = 0
    rng = np.random.default_rng(42)
    with torch.no_grad():
        for _ in range(40):
            field, labels = next(gen)
            for i in range(field.shape[0]):
                dy = rng.integers(-8, 9)
                dx = rng.integers(-8, 9)
                field[i] = torch.roll(field[i], shifts=(int(dy), int(dx)),
                                      dims=(-2, -1))
            if hasattr(sim, "propagate"):
                intensity = sim.propagate(field, phase_mask)
            else:
                intensity = sim(field)
            if use_amax:
                masked = intensity.unsqueeze(1) * bin_masks.unsqueeze(0)
                bins = masked.amax(dim=(-2, -1))
            else:
                bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            correct += (bins.argmax(-1) == labels).sum().item()
            total += len(labels)

    rand_acc = correct / total
    print(f"{'random±8':>8s}  {rand_acc:>10.4f}")

    return results, rand_acc


if __name__ == "__main__":
    main()
