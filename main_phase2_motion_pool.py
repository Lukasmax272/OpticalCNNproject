"""Phase 2b — 4f correlator with motion pooling and Hermitian-symmetric mask.

Replaces the amax pooling and shift curriculum with motion pooling and
a Hermitian-symmetric mask to solve translation invariance structurally.

Run:
    python main_phase2_motion_pool.py [--grid 64] [--shift-range 5] [--iters 5000]
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from four_f_sim import MotionPoolingFourierMask
from utils import load_mnist_generator
from main import build_bin_masks


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64,
                        help="Grid size (default: 64)")
    parser.add_argument("--shift-range", type=int, default=5,
                        help="Max random translation in pixels (applied from step 0)")
    parser.add_argument("--motion-pool-sigma", type=float, default=3.0,
                        help="Sigma for Gaussian motion pool kernel")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Base LR")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (default: 64)")
    args = parser.parse_args()

    grid = args.grid
    batch = args.batch if args.batch > 0 else 64
    effective_lr = args.lr * min(1.0, 64.0 / grid)

    # Calculate kernel size ~6 sigma rounded up to nearest odd integer
    if args.motion_pool_sigma > 1e-5:
        ksize = int(np.ceil(args.motion_pool_sigma * 6))
        if ksize % 2 == 0:
            ksize += 1
    else:
        ksize = 3  # minimum for delta impulse

    sim = MotionPoolingFourierMask(size=grid, sigma=args.motion_pool_sigma, kernel_size=ksize, device=DEVICE)
    sim.to(DEVICE)
    
    gen = load_mnist_generator(batch_size=batch, device=DEVICE, train=True,
                               max_shift=args.shift_range, grid_size=grid)

    # Standard bin layout (margin=0), no enlargement!
    bin_masks = build_bin_masks(grid, DEVICE, margin=0)

    # Optimizer handles the half-spectrum parameters.
    # Weight decay prevents scale drift on the unconstrained complex parameterization.
    optimizer = torch.optim.Adam(sim.parameters(), lr=effective_lr, weight_decay=1e-4)

    accs: list[float] = []
    losses: list[float] = []
    LOG_EVERY = 100

    print(f"Grid: {grid}x{grid}  LR: {effective_lr:.4f}  Batch: {batch}  "
          f"Shift: ±{args.shift_range} (no curriculum)  Sigma: {args.motion_pool_sigma}  Iters: {args.iters}")

    for it in range(args.iters):
        field, labels = next(gen)

        # Forward pass includes FFT -> mask -> IFFT -> |E|^2 -> motion pool
        pooled_intensity = sim.propagate(field)

        # Detection: simple bin sums over standard 3-4-3 layout
        bins = torch.einsum("bhw,khw->bk", pooled_intensity, bin_masks)

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
            
            # Evaluate centered accuracy on a batch of 1000 test images
            sim.eval()
            with torch.no_grad():
                # We can just generate a batch with shift=0 from the training set,
                # or better, create a test generator. For simplicity, just use the train gen with shift=0
                test_gen = load_mnist_generator(batch_size=1000, device=DEVICE, train=False, max_shift=0, grid_size=grid)
                test_field, test_labels = next(test_gen)
                test_intensity = sim.propagate(test_field)
                test_bins = torch.einsum("bhw,khw->bk", test_intensity, bin_masks)
                test_acc = (test_bins.argmax(-1) == test_labels).float().mean().item()
            sim.train()

            print(
                f"iter {it + 1:5d}  loss={recent_loss:.4f}  "
                f"train_acc(last {LOG_EVERY})={recent_acc:.3f}  "
                f"centered_test_acc={test_acc:.3f}",
                flush=True
            )

    # Save mask
    mask_path = f"learned_phase_mask_4f_motion_{grid}.npy"
    full_mask = sim.full_fourier_mask().detach().cpu().numpy()
    np.save(mask_path, full_mask)
    np.save(f"history_phase2b_{grid}p.npy", {"accs": accs, "losses": losses})
    print(f"\nSaved {mask_path}  shape={full_mask.shape}")
    final_acc = sum(accs[-LOG_EVERY:]) / LOG_EVERY
    print(f"Final accuracy (last {LOG_EVERY} iters): {final_acc:.4f}")


if __name__ == "__main__":
    main()
