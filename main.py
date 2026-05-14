"""Phase 1 — direct backprop training of a single-layer diffractive phase mask.

The phase mask is an `nn.Parameter` of shape (64, 64) trained via Adam to
minimize cross-entropy loss between max-normalized detector bin intensities
and MNIST labels. No RL, no env wrapper — the optical model is fully
differentiable through `torch.fft`, so autograd handles everything.

Convergence target: 0.75-0.85 accuracy over ~5000 iterations on a single layer.
The 91.75% reported by Lin/Ozcan is a 5-layer result; don't expect it here.

Run:
    python main.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from physics_sim import OpticalSimulator, MultiLayerD2NN
from utils import load_mnist_generator


# ------------------------------------------------------------------- config
GRID = 64
DATA_BATCH = 64
TOTAL_ITERATIONS = 5000          # ~5 epochs over 60k MNIST at batch 64
LEARNING_RATE = 0.01
LOG_EVERY = 100
DEVICE = torch.device("cpu")

# Optional auxiliary loss that penalises low bin coverage.  Set to True to
# encourage bright-bin routing instead of the default anti-bin (null) routing.
# COVERAGE_LAMBDA controls the trade-off: larger values push more power into
# bins but may hurt discriminability if set too high.  Start with 1.0; if
# coverage stays below 5% after 500 iters, try 5.0 or 10.0.
COVERAGE_LOSS = False
COVERAGE_LAMBDA = 1.0


def build_bin_masks(N: int, device, margin: int = 0) -> torch.Tensor:
    """Three rows of detector bins matching the D²NN literature layout:
      row 0 (top):    classes 0, 1, 2     — 3 bins
      row 1 (middle): classes 3, 4, 5, 6  — 4 bins
      row 2 (bottom): classes 7, 8, 9     — 3 bins

    Row centers at N/6, N/2, 5N/6.  Within each row, bins are evenly spaced
    so the 4-bin middle row doesn't share columns with the 3-bin outer rows.
    Bin size scales with N (half-width = N//16), keeping 1/8 fractional coverage.

    Args:
        N:      grid side length.
        device: torch device.
        margin: extra pixels added to each bin's half-width.  Phase 2 sets
                this to max_shift so the enlarged bin absorbs the full
                translation range.  At large margins the tightest row
                (4 bins at N/5 spacing) will overlap — this is acceptable
                because the 4f mask's pattern selectivity provides
                discrimination; the bins just need to catch the peak.

    Returns a (10, N, N) float tensor; each slice is a binary mask.
    """
    layout = [
        (0, 1/6, 1/4), (1, 1/6, 2/4), (2, 1/6, 3/4),
        (3, 1/2, 1/5), (4, 1/2, 2/5), (5, 1/2, 3/5), (6, 1/2, 4/5),
        (7, 5/6, 1/4), (8, 5/6, 2/4), (9, 5/6, 3/4),
    ]
    half_base = N // 16
    # Cap so the tightest row (4 bins at N/5 spacing) has at most 2x overlap.
    # Max safe half for no overlap = N//10; allow up to N//7 for moderate overlap.
    max_half = N // 7
    half = min(half_base + margin, max_half)
    masks = torch.zeros(10, N, N, dtype=torch.float32, device=device)
    for cls, ry, rx in layout:
        cy, cx = int(ry * N), int(rx * N)
        y0 = max(0, cy - half)
        y1 = min(N, cy + half)
        x0 = max(0, cx - half)
        x1 = min(N, cx + half)
        masks[cls, y0:y1, x0:x1] = 1.0
    return masks


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=3, help="Number of diffractive layers")
    parser.add_argument("--seed", action="store_true", help="Set random seed for reproducibility")
    parser.add_argument("--max-shift", type=int, default=0, help="Max shift pixels for training augmentation")
    parser.add_argument("--iters", type=int, default=5000, help="Total iterations")
    parser.add_argument("--grid", type=int, default=64, help="Grid size")
    parser.add_argument("--intensity-relu", action="store_true", help="Enable intensity threshold ReLU between layers")
    parser.add_argument("--intensity-relu-alpha", type=float, default=0.4, help="Alpha threshold for intensity ReLU")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(42)
        np.random.seed(42)

    # Resolve alpha
    alpha = args.intensity_relu_alpha if args.intensity_relu else None

    # Use MultiLayerD2NN for Phase 1b
    z_total = 0.096
    spacings = [z_total / (args.num_layers + 1)] * (args.num_layers + 1)
    sim = MultiLayerD2NN(
        num_layers=args.num_layers,
        grid_size=args.grid,
        layer_spacings=spacings,
        device=DEVICE,
        intensity_relu_alpha=alpha
    )
    
    gen = load_mnist_generator(batch_size=DATA_BATCH, device=DEVICE, train=True, max_shift=args.max_shift, grid_size=args.grid)
    bin_masks = build_bin_masks(args.grid, DEVICE)

    optimizer = torch.optim.Adam(sim.parameters(), lr=LEARNING_RATE)

    accs: list[float] = []
    losses: list[float] = []

    for it in range(args.iters):
        field, labels = next(gen)

        # Forward pass — gradients flow back through propagate to phase masks.
        intensity = sim(field)                                # (B, N, N)
        bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)  # (B, 10)

        # Max-normalize per Lin/Ozcan and Sheng/Nisar: raw bin sums are O(50-90)
        # and saturate softmax; normalized values land in [0, 1] with one at
        # exactly 1, giving F.cross_entropy a sensible scale to work with.
        normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        loss = F.cross_entropy(normalized, labels)

        if COVERAGE_LOSS:
            # Reward only the correct bin capturing power — not all bins equally.
            # Summing all bins would route light into every detector and flatten
            # the margin; we want a bright target bin and dark non-target bins.
            # CE handles relative suppression; this term handles absolute pull.
            total_power = intensity.sum(dim=(-2, -1))          # (B,)
            correct_bin = bins[torch.arange(len(labels)), labels]  # (B,)
            target_coverage = correct_bin / total_power.clamp(min=1e-8)  # (B,)
            loss = loss - COVERAGE_LAMBDA * target_coverage.mean()

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

    # Save the phase masks
    masks_np = torch.stack([m.detach() for m in sim.phase_masks]).cpu().numpy()
    save_path = f"learned_phase_mask_{args.num_layers}l_{GRID}p.npy"
    np.save(save_path, masks_np)
    np.save(f"history_{args.num_layers}l_{GRID}p.npy", {"accs": accs, "losses": losses})
    print(f"\nSaved {save_path}  shape={masks_np.shape}")
    print(f"Final accuracy (last {LOG_EVERY} iters): {sum(accs[-LOG_EVERY:])/LOG_EVERY:.4f}")


if __name__ == "__main__":
    main()