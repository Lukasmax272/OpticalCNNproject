"""Benchmark: Translation-sweep comparison of all D²NN variants.

This script evaluates trained models on shifted MNIST test sets and produces
 the project headline plot: Accuracy vs. Test-time Shift Magnitude.

Variants:
  - Phase 1a: Single-layer free-space (D²NN)
  - Phase 1b: Multi-layer free-space (D²NN)
  - Phase 1c: Intensity ReLU (Hybrid opto-electronic)
  - Phase 2a: 4f + amax + curriculum (Naive translation attempt)
  - Phase 2b: 4f + motion pooling + Hermitian (Structural invariance)

Usage:
  # Evaluate a specific variant and save results
  python benchmark.py --variant "Phase 2b" --checkpoint learned_phase_mask_4f_motion_64.npy

  # Plot everything in results/
  python benchmark.py --plot-all
"""

import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from physics_sim import OpticalSimulator, MultiLayerD2NN
from four_f_sim import FourFSimulator, MotionPoolingFourierMask
from four_f_hybrid import ODCNNHybrid, build_detector_masks_200
from utils import load_mnist_generator
from main import build_bin_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sweep_shifts(variant_name, forward_fn, grid, shift_range=7, n_batches=40,
                 batch_size=256, amplitude_encode=False, upsample=True):
    """Evaluate a model across a range of horizontal pixel shifts."""
    print(f"Sweeping shifts for {variant_name}...")
    results = {}
    shifts = range(-shift_range, shift_range + 1)
    
    # We use a fixed test generator
    gen = load_mnist_generator(batch_size, DEVICE, train=False, grid_size=grid,
                               amplitude_encode=amplitude_encode, upsample=upsample)
    
    for s in shifts:
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(n_batches):
                field, labels = next(gen)
                # Apply shift (horizontal only as per prompt, but dy=0)
                if s != 0:
                    field = torch.roll(field, shifts=(0, s), dims=(-2, -1))
                
                intensity = forward_fn(field)
                
                # Detection (bins are handled inside forward_fn or by its output)
                # For consistency, we assume forward_fn returns normalized bin sums (B, 10)
                # OR raw bin sums. We'll handle both.
                if intensity.ndim == 2 and intensity.shape[1] == 10:
                    bins = intensity
                else:
                    # If it returns intensity image, we need binning
                    # This shouldn't happen with the current plan but good to be safe
                    raise ValueError("forward_fn must return bin sums (B, 10)")
                
                correct += (bins.argmax(-1) == labels).sum().item()
                total += len(labels)
        
        acc = correct / total
        results[s] = acc
        print(f"  shift={s:+2d}  acc={acc:.4f}")
        
    return results


def plot_translation_sweep(results_dir="results", output_name="translation_invariance_mnist"):
    """Aggregate all .json results and produce the headline plot."""
    files = [f for f in os.listdir(results_dir) if f.endswith("_sweep.json")]
    if not files:
        print(f"No results found in {results_dir}")
        return

    plt.figure(figsize=(10, 6), dpi=300)
    
    # Legend names with descriptions
    descriptions = {
        "Phase 1a": "Phase 1a: Single-layer free-space",
        "Phase 1b": "Phase 1b: 3-layer D²NN",
        "Phase 1c": "Phase 1c: 3-layer D²NN + Intensity ReLU",
        "Phase 2a": "Phase 2a: 4f + amax pooling",
        "Phase 2b": "Phase 2b: 4f + motion pooling",
        "Phase 2c": "Phase 2c: ODCNN hybrid",
    }

    # Colors for variants (using base names as keys)
    colors = {
        "Phase 1a": "#3b82f6", # Blue
        "Phase 1b": "#10b981", # Green
        "Phase 1c": "#8b5cf6", # Purple
        "Phase 2a": "#ef4444", # Red
        "Phase 2b": "#f97316", # Orange
        "Phase 2c": "#facc15", # Yellow
    }

    found_variants = []
    for f in files:
        with open(os.path.join(results_dir, f), 'r') as j:
            data = json.load(j)
            found_variants.append(data)
            
    # Sort by order list (using base names)
    order = ["Phase 1a", "Phase 1b", "Phase 1c", "Phase 2a", "Phase 2b", "Phase 2c"]
    found_variants.sort(key=lambda x: order.index(x['name']) if x['name'] in order else 99)

    for data in found_variants:
        name = data['name'] # This is the base name, e.g., "Phase 2b"
        label = descriptions.get(name, name)
        res = data['results']
        # JSON keys are strings, convert back to ints
        shifts = sorted([int(k) for k in res.keys()])
        accs = [res[str(s)] for s in shifts]
        
        color = colors.get(name, None)
        plt.plot(shifts, accs, marker='o', markersize=4, label=label, color=color, linewidth=2)

    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(0, 1.05)
    plt.xlabel("Horizontal Pixel Shift", fontsize=12)
    plt.ylabel("MNIST Test Accuracy", fontsize=12)
    plt.title("Translation invariance of optical architectures (MNIST)", fontsize=14, pad=15)
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_name}.png")
    plt.savefig(f"{output_name}.pdf")
    print(f"Plot saved to {output_name}.png and .pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, help="Variant name (e.g. 'Phase 1a')")
    parser.add_argument("--checkpoint", type=str, help="Path to .npy weights")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--intensity-relu", action="store_true")
    parser.add_argument("--relu-alpha", type=float, default=0.4)
    parser.add_argument("--shift-range", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--motion-pool-sigma", type=float, default=0.0,
                        help="Motion pool sigma for Phase 2c (0=off)")
    parser.add_argument("--plot-all", action="store_true")
    args = parser.parse_args()

    if args.plot_all:
        plot_translation_sweep(args.output_dir)
        return

    if not args.variant or not args.checkpoint:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and binning based on variant
    grid = args.grid
    bin_masks = None
    
    if "Phase 1" in args.variant or args.variant == "Phase 1a" or args.variant == "Phase 1b":
        # Multi-layer D2NN setup
        z_total = 0.096
        spacings = [z_total / (args.num_layers + 1)] * (args.num_layers + 1)
        model = MultiLayerD2NN(
            num_layers=args.num_layers,
            grid_size=grid,
            layer_spacings=spacings,
            device=DEVICE,
            intensity_relu_alpha=args.relu_alpha if args.intensity_relu else None
        )
        # Load weights
        masks = np.load(args.checkpoint)
        if masks.ndim == 2: # handle single layer save
            masks = masks[np.newaxis, ...]
        with torch.no_grad():
            for i, m in enumerate(masks):
                model.phase_masks[i].copy_(torch.from_numpy(m))
        
        bin_masks = build_bin_masks(grid, DEVICE)
        
        def forward_fn(field):
            intensity = model(field)
            bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            return bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)

    elif args.variant == "Phase 2a":
        # 4f + amax
        sim = FourFSimulator(size=grid, device=DEVICE)
        mask = torch.from_numpy(np.load(args.checkpoint)).to(DEVICE)
        # Use enlarged bins for 2a (as per instructions and training)
        bin_masks = build_bin_masks(grid, DEVICE, margin=8) # matching main_phase2.py default
        
        def forward_fn(field):
            intensity = sim.propagate(field, mask)
            masked = intensity.unsqueeze(1) * bin_masks.unsqueeze(0)
            bins = masked.amax(dim=(-2, -1))
            return bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)

    elif args.variant == "Phase 2b":
        # 4f + motion pooling
        # Note: MotionPoolingFourierMask expects sigma/ksize. 
        # We assume defaults: sigma=3.0, ksize=19
        model = MotionPoolingFourierMask(size=grid, sigma=3.0, kernel_size=19, device=DEVICE)
        # 2b saves the FULL mask
        full_mask = torch.from_numpy(np.load(args.checkpoint)).to(DEVICE)
        
        bin_masks = build_bin_masks(grid, DEVICE, margin=0) # standard bins
        
        def forward_fn(field):
            # We can pass the full mask directly to propagate
            pooled_intensity = model.propagate(field, phase_mask=full_mask)
            bins = torch.einsum("bhw,khw->bk", pooled_intensity, bin_masks)
            return bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    
    elif args.variant == "Phase 2c":
        # ODCNN hybrid: tiled 4f + complex-modulation D²NN
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = ODCNNHybrid(
            grid_size=ckpt["grid_size"],
            num_kernels=ckpt["num_kernels"],
            kernel_size=ckpt["kernel_size"],
            d2nn_layers=ckpt["d2nn_layers"],
            d2nn_z_total=ckpt["d2nn_z_total"],
            pixel_size=ckpt["pixel_size"],
            motion_pool_sigma=ckpt.get("motion_pool_sigma", args.motion_pool_sigma),
        ).to(DEVICE)
        # Restore weights
        with torch.no_grad():
            model.tiled_mask.spatial_kernels.copy_(ckpt["spatial_kernels"].to(DEVICE))
            for i, al in enumerate(ckpt["amplitude_logits"]):
                model.d2nn.amplitude_logits[i].copy_(al.to(DEVICE))
            for i, pm in enumerate(ckpt["phase_masks"]):
                model.d2nn.phase_masks[i].copy_(pm.to(DEVICE))
        model.eval()
        grid = ckpt["grid_size"]

        bin_masks = build_detector_masks_200(grid, DEVICE)

        def forward_fn(field):
            intensity = model(field)
            bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            return bins

    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    # Phase 2c uses amplitude encoding and NO upsampling (matches main_phase2_hybrid.py)
    use_amplitude = (args.variant == "Phase 2c")
    use_upsample = (args.variant != "Phase 2c")

    # Run sweep
    results = sweep_shifts(args.variant, forward_fn, grid, shift_range=args.shift_range,
                           amplitude_encode=use_amplitude, upsample=use_upsample)
    
    # Save to JSON
    save_data = {
        "name": args.variant,
        "checkpoint": args.checkpoint,
        "grid": grid,
        "results": results
    }
    safe_name = args.variant.replace(" ", "_").lower()
    save_path = os.path.join(args.output_dir, f"{safe_name}_sweep.json")
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
