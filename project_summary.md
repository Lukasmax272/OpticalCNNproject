# D²NN Optical MNIST Project: Summary and Retrospective

## Motivation
The primary motivation of this project is to explore and simulate differentiable optical neural networks (ONNs) capable of classifying MNIST digits *in silico*. Traditional deep learning relies heavily on power-hungry electronic processors (GPUs/TPUs). Optical neural networks offer the promise of computation at the speed of light with near-zero energy consumption for the forward pass, relying on the physical propagation of light waves through diffractive phase masks. 

A central focus of this project is teaching and demonstrating the progression from a simple, spatially-rigid all-optical network to a robust, translation-invariant architecture. In typical diffractive deep neural networks (D²NNs), a slight shift in the input image drastically degrades performance because the network acts essentially as a rigid spatial filter. Overcoming this limitation to achieve true shift-invariance (a property inherently possessed by electronic convolutional neural networks) is the project's defining technical arc.

## Design and Architecture
The project is fundamentally driven by simulating coherent light propagation using wave optics (the angular spectrum method) in a fully differentiable PyTorch environment. Because the simulation uses `torch.fft` and complex-valued elementwise operations, the entire optical pipeline is differentiable, allowing end-to-end training via direct backpropagation with Adam.

The architecture was developed through two distinct phases, with each step preserved as a checkpoint to benchmark progress:

### Phase 1: Free-Space Diffractive Neural Networks
- **1a. Single-Layer Baseline:** A basic all-optical ONN using a single learnable phase mask. This establishes a foundational baseline, typically yielding 80–85% accuracy on centered MNIST digits.
- **1b. Multi-Layer D²NN:** Introduces depth. By stacking multiple learnable phase masks separated by free-space propagation distances, the model captures more complex spatial transformations, consistently pushing accuracy above 90%.
- **1c. Hybrid Opto-Electronic Nonlinearity:** Incorporates an intensity-threshold ReLU between diffractive layers. This simulates a "detect-and-re-emit" plane, injecting non-linearity without requiring specialized non-linear optical materials.

### Phase 2: Translation-Invariant ONNs
- **2a. Naive 4f Correlator:** Replaces free-space propagation with a 4f optical correlator setup using a learnable Fourier-plane phase mask, paired with a naive shift curriculum. This served as a baseline to demonstrate the failure of standard pooling (like `amax`) in achieving robust shift invariance.
- **2b. Motion Pooling + Hermitian Symmetry:** A breakthrough step replacing `amax` with a differentiable Gaussian "motion pooling" kernel applied to the output intensity. Combined with enforcing Hermitian symmetry on the Fourier mask (guaranteeing a real-valued spatial impulse response), this model achieved ≥95% accuracy while remaining highly robust to ±5 pixel shifts.
- **2c. ODCNN Hybrid:** A state-of-the-art hybrid optical diffractive convolutional neural network (ODCNN). It employs a tiled multi-kernel 4f correlator stage for feature extraction, followed by a multi-layer D²NN classifier. Operating on a high-resolution (200×200) grid, it achieved the target of ≥97% accuracy on shifted MNIST, demonstrating equivalence to electronic CNNs.

## Results and Benchmarks
The progression of architectures yields a clear narrative of improvement, particularly evaluated against the dual metrics of **baseline accuracy** and **translation invariance**:
* **Baseline D²NNs** (1a/1b) performed well on centered digits but experienced catastrophic accuracy drops (often below 20%) when inputs were shifted by even a few pixels.
* **Phase 2b (Motion Pooling)** maintained >90% accuracy across a ±5 pixel shift sweep, proving that appropriate pooling over diffraction spots enables spatial generalization.
* **Phase 2c (ODCNN)** dominated both metrics, achieving top-tier classification accuracy (≥97%) while matching the flat, robust translation-invariance curve of electronic CNNs.

## Technical Challenges and Solutions
Simulating physical optics *in silico* for machine learning presented several unique hurdles:
1. **Numerical Stability of Phase:** The `exp(i·kz)` transfer function in free space creates massive phase accumulations (>10⁶ rad). This destroyed `float32` precision. The solution was implementing **carrier-phase subtraction**, cleanly removing the on-axis carrier wave before exponentiation.
2. **Vanishing Gradients at Detection:** Raw detector bin intensities are physically large. Feeding these directly to `F.cross_entropy` saturated the softmax and halted training. We solved this by implementing **max-normalization** of the bin intensities prior to the loss function, ensuring stable gradients.
3. **Reinforcement Learning Failures:** Early attempts to optimize the phase masks using PPO (Proximal Policy Optimization) failed completely. The continuous action space (e.g., 4096-dimensional masks) caused the independent Gaussian policy's importance ratios to compound exponentially, destroying the surrogate clip interval. Replacing RL with **differentiable physics and Adam** was critical to success.
4. **Memory Bottlenecks (Phase 2c):** Scaling to the 200×200 grid required for ODCNN exhausted standard VRAM allocations during autograd. We had to heavily refactor the data pipeline and intermediate tensor allocations to make the model trainable on standard cloud GPUs (e.g., Colab T4).

## Areas for Improvement
While the project successfully simulates an advanced translation-invariant optical network, several avenues remain for future exploration:
1. **All-Optical Kerr Nonlinearities:** Phase 1c relies on an opto-electronic threshold (detect, threshold in electronics, re-emit). Exploring simulated Kerr effect materials would allow for true, continuous all-optical non-linearity without leaving the optical domain.
2. **Hardware Implementation:** The current parameters (wavelength, pixel pitch, propagation distance) are physically plausible, but transitioning the continuous unbounded phase masks `[0, 2π]` to discrete quantization levels (e.g., for manufacturing via photolithography or spatial light modulators) would require quantization-aware training.
3. **Phase 5 (Autonomous MARL):** Although currently out of scope, using Multi-Agent Reinforcement Learning to discover novel optical topologies—rather than just tuning the weights of a human-designed 4f or D²NN topology—remains an exciting frontier.
