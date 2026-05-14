"""Evaluate the learned phase mask on the MNIST test set."""

import math
import torch
import numpy as np
from physics_sim import OpticalSimulator
from utils import load_mnist_generator
from main import build_bin_masks

DEVICE = torch.device("cpu")

sim = OpticalSimulator(size=64, device=DEVICE)
gen = load_mnist_generator(batch_size=256, device=DEVICE, train=False)
bin_masks = build_bin_masks(64, DEVICE)

phase_mask = torch.from_numpy(np.load("learned_phase_mask.npy")).to(DEVICE)

correct = 0
total = 0
with torch.no_grad():
    for _ in range(40):  # ~10k test digits
        field, labels = next(gen)
        intensity = sim.propagate(field, phase_mask)
        bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
        correct += (bins.argmax(-1) == labels).sum().item()
        total += len(labels)

print(f"Test accuracy: {correct / total:.4f}  ({correct}/{total})")
