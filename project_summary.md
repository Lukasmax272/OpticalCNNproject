# D²NN Optical MNIST: Summary and Retrospective

## Motivation

Optical neural networks compute by sending light through stacks of diffractive phase masks. Once the masks are fabricated, the forward pass costs essentially nothing — the wave propagates at c and the "computation" is just physics. That makes ONNs an attractive alternative to electronic inference, at least for problems that fit the architecture.

The catch is that a plain diffractive network is a rigid spatial filter. Shift the input a few pixels and accuracy collapses. Electronic CNNs sidestep this with weight sharing and pooling; getting a comparable shift-invariance property out of a coherent optical system is the central problem this project takes on.

## Architecture

Everything is simulated with the angular spectrum method in PyTorch. Because `torch.fft` and complex elementwise ops are differentiable, the full pipeline — phase masks, free-space propagation, optional nonlinearity, detector binning — trains end-to-end with Adam. The project went through two phases, with each variant kept as a checkpoint.

### Phase 1: free-space D²NNs

**1a — single-layer baseline.** One learnable phase mask, free-space propagation, intensity binning. 80–85% on centered MNIST. Sets the floor.

**1b — multi-layer D²NN.** Several phase masks separated by short propagation distances. Depth helps; accuracy clears 90%.

**1c — hybrid opto-electronic nonlinearity.** An intensity-threshold ReLU between layers, modeling a detect-and-reemit plane. A cheap stand-in for true optical nonlinearity, and a small but real accuracy bump.

### Phase 2: translation-invariant ONNs

**2a — naive 4f correlator.** Free-space propagation swapped for a 4f setup with a learnable Fourier-plane mask, trained on a shift curriculum, pooled with `amax`. The point of keeping this checkpoint is to show that the obvious design *doesn't* deliver shift invariance.

**2b — motion pooling + Hermitian symmetry.** Replace `amax` with a differentiable Gaussian "motion pooling" kernel over the output intensity, and constrain the Fourier mask to be Hermitian-symmetric so its impulse response is real. Together these gave ≥95% accuracy holding within a ±5 pixel shift window — the first version that actually generalizes spatially.

**2c — ODCNN hybrid.** A tiled multi-kernel 4f stage feeding a multi-layer D²NN classifier, on a 200×200 grid. ≥97% on shifted MNIST, with a flat shift-vs-accuracy curve comparable to a small electronic CNN.

## Results

The headline numbers, evaluated on centered MNIST and on a ±5 pixel shift sweep:

- 1a and 1b handle centered MNIST cleanly but fall below 20% once shifts get past a couple of pixels. Confirms the rigid-filter problem.
- 2b sustains >90% across the shift sweep. Motion pooling over diffraction spots is doing the work.
- 2c hits ≥97% centered *and* across the shift range — the only model that beats both metrics simultaneously.

## What broke, and what fixed it

A few problems were genuinely interesting:

**Phase wraps killed float32.** The free-space transfer function exp(i·kz) accumulates phases on the order of 10⁶ rad over realistic propagation distances at visible wavelengths. Float32 doesn't have the precision for that, and the transfer function turns into noise. Fix: subtract the on-axis carrier phase before exponentiation, so only the off-axis residual goes through `exp`. After that the simulation is well-conditioned.

**Detector intensities saturated the softmax.** Raw bin intensities are large and have a wide dynamic range; piping them straight into `F.cross_entropy` makes the softmax effectively one-hot and gradients vanish. Max-normalizing the bins before the loss restored gradient flow. In retrospect this is the optical analogue of logit scaling and should have been obvious earlier.

**PPO didn't work, and wasn't going to.** The first attempt at training the masks used PPO with a 4096-dimensional Gaussian action. With actions that high-dimensional, the per-step importance ratio is a product of thousands of per-dimension ratios; the clip interval is shredded on the first batch. Switching to differentiable physics and plain Adam isn't just simpler — it's what the problem actually wants. The masks are continuous and the simulation is differentiable; there's no reason to throw a policy-gradient method at it.

**VRAM at 200×200.** ODCNN training on a 200×200 grid blew through Colab T4 memory the first few attempts. The fix was the usual: rework the data pipeline to drop unnecessary intermediates, recompute where cheaper than storing, batch size tuned by hand.

## Open questions

A few directions worth pursuing:

- **All-optical nonlinearity.** The threshold in 1c is a cheat — it leaves the optical domain. A simulated Kerr-effect material would put the nonlinearity back inside the optical pipeline. Whether it trains as cleanly is an open question.
- **Quantization-aware training.** Phase masks here are continuous on [0, 2π]. Real SLMs and lithographically fabricated masks are quantized to a handful of levels. The simulation-to-hardware gap probably costs accuracy; it would be worth measuring how much, and whether QAT closes it.
- **Topology search.** Everything here optimizes weights inside a human-designed topology — 4f, D²NN, ODCNN. Using a search method to discover the topology itself is much harder, and probably the next interesting problem.
