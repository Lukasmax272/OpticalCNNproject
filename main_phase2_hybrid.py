"""Phase 2c: ODCNN Hybrid Training — faithful Sensors 2023 reproduction.

Diverges from project conventions in three ways (see four_f_hybrid.py docstring):
  1. Amplitude encoding (E_in = pixel + 0j)
  2. Complex modulation (t = α·exp(jφ))
  3. MSE on softmax outputs (not NSCE)

These divergences are documented because they match the paper exactly.

Defaults match the paper:
  - 16 kernels of 9×9, 4×4 tiled on 200×200
  - 5-layer D²NN classifier, 200×200
  - Adam, lr=1e-3, 10000 iterations, batch 64
  - MSE(softmax(bins), one_hot(label))

Usage on Colab:
    python main_phase2_hybrid.py

If T4 memory is tight:
    python main_phase2_hybrid.py --batch-size 32 --amp
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

from four_f_hybrid import ODCNNHybrid, build_detector_masks_200
from utils import load_mnist_generator


def main():
    parser = argparse.ArgumentParser(description="Phase 2c: ODCNN Hybrid Training")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--num-kernels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--d2nn-layers", type=int, default=5)
    parser.add_argument("--d2nn-z-total", type=float, default=0.3,
                        help="Total D²NN propagation distance (m)")
    parser.add_argument("--pixel-size", type=float, default=8e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--motion-pool-sigma", type=float, default=0.0,
                        help="Optional Gaussian motion pooling sigma (0 = off)")
    parser.add_argument("--shift-range", type=int, default=0,
                        help="Max random translation in pixels (0 = centered only)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision (torch.amp.autocast)")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--save-prefix", type=str, default="odcnn_hybrid",
                        help="Prefix for saved checkpoint files")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(42)
        np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Model ---
    model = ODCNNHybrid(
        grid_size=args.grid_size,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        d2nn_layers=args.d2nn_layers,
        d2nn_z_total=args.d2nn_z_total,
        pixel_size=args.pixel_size,
        motion_pool_sigma=args.motion_pool_sigma,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Data: amplitude encoding (Divergence 1) ---
    train_gen = load_mnist_generator(
        batch_size=args.batch_size,
        device=device,
        train=True,
        max_shift=args.shift_range,
        grid_size=args.grid_size,
        upsample=False,
        amplitude_encode=True,
    )

    test_gen = load_mnist_generator(
        batch_size=args.batch_size,
        device=device,
        train=False,
        max_shift=0,
        grid_size=args.grid_size,
        upsample=False,
        amplitude_encode=True,
    )

    # --- Detector masks ---
    bin_masks = build_detector_masks_200(args.grid_size, device)

    # --- Optimizer: Adam, no weight decay (paper doesn't mention it) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training ---
    LOG_EVERY = 50
    TEST_EVERY = 500
    TEST_BATCHES = 16  # 16 × batch_size samples for test eval

    accs = []
    losses = []

    print(f"\nODCNN Hybrid: {args.grid_size}×{args.grid_size}, "
          f"{args.num_kernels} kernels of {args.kernel_size}×{args.kernel_size}")
    print(f"D²NN: {args.d2nn_layers} layers, complex modulation")
    print(f"Loss: MSE on softmax (Sensors 2023 eq. 5)")
    print(f"Batch: {args.batch_size}, Iters: {args.iterations}, LR: {args.lr}")
    if args.shift_range > 0:
        print(f"Shift augmentation: ±{args.shift_range} px")
    if args.motion_pool_sigma > 0:
        print(f"Motion pooling: σ={args.motion_pool_sigma}")
    print()

    # Amplitude encoding sanity check on first batch
    field_check, _ = next(train_gen)
    nonzero_mask = field_check.abs() > 0
    if nonzero_mask.any():
        amp_std = field_check.abs()[nonzero_mask].std().item()
    else:
        amp_std = 0.0
    print(f"Amplitude encoding check: E_in.abs().std() (nonzero pixels) = {amp_std:.4f} "
          f"({'OK -- varying amplitude' if amp_std > 0.1 else 'WARNING: uniform amplitude!'})")
    # Put it back by remaking the generator (the check consumed one batch)
    train_gen = load_mnist_generator(
        batch_size=args.batch_size, device=device, train=True,
        max_shift=args.shift_range, grid_size=args.grid_size,
        upsample=False, amplitude_encode=True,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    for it in range(args.iterations):
        field, labels = next(train_gen)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                intensity = model(field)
                bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
                # MSE on softmax (Divergence 3, paper eq. 5)
                probs = F.softmax(bins, dim=-1)
                target = F.one_hot(labels, num_classes=10).float()
                loss = F.mse_loss(probs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            intensity = model(field)
            bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            # MSE on softmax (Divergence 3, paper eq. 5)
            probs = F.softmax(bins, dim=-1)
            target = F.one_hot(labels, num_classes=10).float()
            loss = F.mse_loss(probs, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc = (bins.argmax(-1) == labels).float().mean().item()
        accs.append(acc)
        losses.append(loss.item())

        if (it + 1) % LOG_EVERY == 0:
            recent_acc = sum(accs[-LOG_EVERY:]) / LOG_EVERY
            recent_loss = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            print(f"iter {it + 1:5d}  loss={recent_loss:.4f}  "
                  f"train_acc(last {LOG_EVERY})={recent_acc:.3f}")

        # Periodic test evaluation
        if (it + 1) % TEST_EVERY == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(TEST_BATCHES):
                    tf, tl = next(test_gen)
                    ti = model(tf)
                    tb = torch.einsum("bhw,khw->bk", ti, bin_masks)
                    correct += (tb.argmax(-1) == tl).sum().item()
                    total += len(tl)
            test_acc = correct / total
            print(f"  >>> TEST accuracy ({total} samples): {test_acc:.4f}")
            model.train()

        # Gradient sanity check at iteration 10
        if it == 9:
            amp_grad = model.d2nn.amplitude_logits[0].grad
            phi_grad = model.d2nn.phase_masks[0].grad
            if amp_grad is not None and phi_grad is not None:
                print(f"  Gradient check: amp_logits[0].grad.norm={amp_grad.norm():.6f}, "
                      f"phase_masks[0].grad.norm={phi_grad.norm():.6f}")
            else:
                print("  WARNING: gradients are None — check model connectivity")

    # --- Final test ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(TEST_BATCHES * 4):
            tf, tl = next(test_gen)
            ti = model(tf)
            tb = torch.einsum("bhw,khw->bk", ti, bin_masks)
            correct += (tb.argmax(-1) == tl).sum().item()
            total += len(tl)
    final_test_acc = correct / total
    print(f"\nFinal test accuracy ({total} samples): {final_test_acc:.4f}")
    print(f"Final train accuracy (last {LOG_EVERY} iters): "
          f"{sum(accs[-LOG_EVERY:])/LOG_EVERY:.4f}")

    # --- Save checkpoint ---
    os.makedirs("results", exist_ok=True)
    ckpt = {
        "spatial_kernels": model.tiled_mask.spatial_kernels.detach().cpu(),
        "amplitude_logits": [m.detach().cpu() for m in model.d2nn.amplitude_logits],
        "phase_masks": [m.detach().cpu() for m in model.d2nn.phase_masks],
        "grid_size": args.grid_size,
        "d2nn_layers": args.d2nn_layers,
        "num_kernels": args.num_kernels,
        "kernel_size": args.kernel_size,
        "d2nn_z_total": args.d2nn_z_total,
        "pixel_size": args.pixel_size,
        "motion_pool_sigma": args.motion_pool_sigma,
        "final_test_acc": final_test_acc,
        "accs": accs,
        "losses": losses,
    }
    save_path = f"results/{args.save_prefix}.pt"
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
