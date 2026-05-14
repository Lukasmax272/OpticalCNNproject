# Research Reference — D²NN Optical MNIST Project

Synthesized content from the five papers grounding this project. Use this as the source of truth when citing architectures, equations, accuracy results, or design rationale. The PDFs themselves cannot be uploaded; this is the agent-facing summary.

Cite originating paper and equation in docstrings (e.g., `"""Motion pooling per Sadeghzadeh et al., IEEE Access 2021, eq. 17."""`).

---

## Paper 1: Lin, Ozcan et al., *Science* 2018 — original D²NN

**Title:** All-Optical Machine Learning Using Diffractive Deep Neural Networks

**Architecture:** 5-layer phase-only transmissive D²NN, 3D-printed for THz spectrum. Each point on a layer acts as a neuron — a secondary wave source whose amplitude and phase are set by the product of the input field and the local complex transmission coefficient. Successive layers connected via Rayleigh-Sommerfeld diffraction.

**Encoding:** Input digits encoded into the **amplitude** of the input field. (Note: the current codebase uses **phase** encoding via `exp(i·π·pixel)`, which is a deliberate different choice we are keeping.)

**Output:** Ten detector regions at the output plane, one per digit. Classification criterion is the detector with maximum optical signal, which also serves as the training loss (energy maximization at the correct detector).

**Result:** 91.75% MNIST classification accuracy on 10,000-image test set. Experimentally validated with 50 3D-printed handwritten digits achieving 88% match between numerical simulation and physical experiment.

**Key insight for this project:** Depth matters. The 91.75% result requires 5 phase-only layers. The receptive field of any one layer is set by the axial spacing between layers, the wavelength, and the detection SNR — not chosen freely.

---

## Paper 2: Sheng & Nisar, *Micromachines* 2024 — Integrated D²NN (ID²NN)

**Title:** Integrated photonic D²NN on silicon-on-insulator

**Architecture:** 5 layers × 300 neurons each, etched slots in silicon-on-insulator (SOI). Slot length (0.2–2.5 µm) at fixed width (0.14 µm) imposes phase delays 0 → 2π with >90% transmission. Wavelength 1550 nm in free space, 300 µm between layers.

**Encoding:** MNIST binarized (0/1) and downsampled to N=100 pixels (10×10), flattened. Pixel value 1 → light propagates, pixel value 0 → blocked.

**Forward propagation (eq. 1):** Rayleigh-Sommerfeld formula:

```
w_i^l(x, y, z) = ((z - z_i) / r²) · (1/(2πr) + 1/(jλ_g)) · exp(j·2πr/λ_g)
```

where `r = sqrt((x-x_i)² + (y-y_i)² + (z-z_i)²)`, λ_g is the wavelength in the guided medium.

**Neuron transmission coefficient (eq. 3):**
```
t_i^l(x_i, y_i, z_i) = α_i^l · exp(j·φ_i^l)
```

with α ∈ (0, 1), φ ∈ (0, 2π).

**Loss function (eq. 5) — Normalized Softmax Cross-Entropy (NSCE):** This is the loss the current codebase already uses:

```
min over φ_i^l of: -Σ_n g̃_n · log(exp(s̃_n) / Σ exp(s̃_n))
```

where `s̃_n = s_n / max(s_n)` is the max-normalized detector intensity and g̃_n is the target distribution.

**Result:** 91.6% MNIST accuracy with N=100 features. 100 features was the optimal trade-off between accuracy and footprint.

**Key insight for this project:** Validates the NSCE loss already in use. Confirms [0, 2π] phase range. Confirms 5-layer depth as the published convergence point for ~91% MNIST accuracy.

---

## Paper 3: Sensors 2023 — Optical Diffractive CNN (ODCNN)

**Title:** Optical convolutional neural network combining 4f system and D²NN

**Architecture:** Hybrid two-stage all-optical pipeline:
1. **4f optical convolution** with tiled multi-kernel phase mask in Fourier plane.
2. **5-layer diffractive D²NN classifier** receiving the convolved feature maps.

**Tiled multi-kernel mask (eq. 1):**
```
PSF(x, y) = Σ_m Σ_n Kernel_{(m-1)N+n}(x, y) ∗ δ(x - Δx, y - Δy)
```

After parameter sweeping, the paper settled on **16 kernels of 9×9 pixels each, arranged in a 4×4 spatial grid on a 200×200 phase mask**. Spacing (Δx, Δy) tuned to prevent crosstalk between the 16 feature maps. The output is a 200×200 plane containing 16 spatially-tiled feature maps.

**Phase filter only:** The Fourier-plane mask modifies phase of each frequency component without affecting amplitude, giving high optical efficiency.

**Intensity-domain threshold ReLU (eq. 7):**
```
f(x) = x  if x ≥ α · 255
f(x) = 0  otherwise
```

with α = 0.4 used for Fashion-MNIST. Applied after the 4f stage on detected intensity. The threshold formulation (instead of standard ReLU) is because light intensity cannot be negative, so traditional ReLU truncation is ill-defined. Realizable physically via sCMOS camera curve adjustment or photorefractive crystals (SBN:60).

**Comparison results on MNIST:**
- Pure D²NN baseline (5 layers, no conv front-end): **92.35%**
- ODCNN (4f tiled conv + 5-layer D²NN): **97.32%**
- Adam optimizer, MSE loss, 10,000 iterations, batch size 64

**Key insights for this project:**
1. The tiled multi-kernel 4f → multi-layer D²NN architecture is the Phase 2c target. Strongest published all-optical MNIST architecture.
2. The intensity threshold ReLU is the right nonlinearity form for this codebase — physically realizable, well-defined for non-negative intensity, used between optical stages.
3. 4f shift-equivariance followed by D²NN shift-tolerance is what gives the architecture translation robustness.

---

## Paper 4: Sadeghzadeh et al., *IEEE Access* 2021 — motion pooling

**Title:** Free-Space Optical Neural Network Based on Optical Nonlinearity and Pooling Operations

**Architecture:** 4f optical correlator with saturable absorption nonlinearity between 4f and detector, and a Gaussian mask in the Fourier plane implementing "motion pooling."

**Square-law photodetection (eq. 18):**
```
η(r) = |r|²
```

where r is the complex field at the detector. Formalizes the photodetector as a free nonlinearity in any ONN. The current codebase already implements this via `out.real.square() + out.imag.square()`.

**Motion pooling (eq. 17):**
```
Z(x, y) = exp(-(x - w/2)² / (2σ²))
```

A narrow Gaussian mask placed in the Fourier plane of a 4f system. Because the Gaussian is an eigenfunction of the Fourier transform, multiplying the spectrum by a narrow Gaussian is equivalent to convolving the spatial image with a wide Gaussian — deliberately blurring features across their neighborhoods. This is the operation that confers translation invariance.

For MNIST simulation in the paper: Gaussian filter mask with σ = 10 in the Fourier plane (Section 7 of Supplement 1). Pixel pitch 2.5 µm.

**Translation invariance results (Table 5):**

Compares OPL1-SA-MotionPool vs OPL1-SA-AvgPool on translated MNIST test images when training images were not translated:

| Test shift | MotionPool | AvgPool |
|---|---|---|
| ±1, ±2 px | similar accuracy | similar accuracy |
| ±3 to ±6 px | considerably higher | drops sharply |

When training images are shift-augmented (±3, ±5 pixels), MotionPool generalizes to test shifts up to ±7 pixels with 97.66–97.98% MNIST accuracy.

**Best centered MNIST results in the paper:** 99.25% with optical average pooling, 99.01% with motion pooling. The headline number is translation generalization, not centered accuracy.

**Saturable absorption nonlinearity (eqs. 4-5):**
```
E_P,out = exp(-α₀/2 / (1 + E_P,in²)) · E_P,in
```

The paper also models saturable absorption as an alternative optical nonlinearity, with both phenomenological and four-level atomic models. For this project we use the threshold ReLU from Paper 3 instead — it's simpler to simulate and physically realizable via camera-curve adjustment.

**Key insights for this project:**
1. Motion pooling is the published, gradient-friendly solution to the translation problem. The Phase 2a code's `amax` pooling has zero gradient outside the argmax pixel — catastrophic for training. Motion pooling (Gaussian blur on intensity, or equivalently a Gaussian Fourier-plane mask) is differentiable everywhere with σ as the explicit translation-invariance budget.
2. Motion pooling can be applied at either plane — Gaussian in Fourier space and Gaussian convolution in real space are mathematically equivalent (FFT eigenfunction). This codebase implements it as a fixed Gaussian conv on the intensity image after detection, because that's the cleanest insertion point and the kernel can be a non-trainable buffer.
3. The shift curriculum in Phase 2a is a workaround for amax's gradient pathology. With motion pooling it becomes unnecessary — train with full augmentation from step 0.

---

## Paper 5: Schultz et al., *IEEE Photonics Benelux* 2021 — practical 4f

**Title:** Optical 4F Correlator for Acceleration of Convolutional Neural Networks

**Architecture:** Off-the-shelf optical 4f correlator: 633nm He-Ne laser, Keplerian beam expander, two LCoS SLMs (image on SLM1, Fourier-domain kernel on SLM2), CMOS camera at output plane.

**Key training trick — Fourier-domain kernel training with Hermitian symmetry:**

The Fourier transform of a real-valued spatial kernel is Hermitian-symmetric: `K̂(-k) = K̂*(k)`. Training only the positive half of the kernel (plus DC) halves the parameter count and guarantees the spatial kernel is real-valued. The full kernel is reconstructed before being displayed on the SLM by "unfolding" (mirroring with conjugation).

**Why train in the Fourier domain at all:**
- Saves the FFT of the kernel at inference time
- Acts as a powerful regularizer on the Fourier landscape
- Maps directly to what the physical SLM displays (Fourier-plane mask)

**Result:** 91% MNIST best-case accuracy on the optical correlator (vs 98.2% for an electronic CNN with the same architecture). Discrepancy attributed to camera flickering, alignment issues, and FC-layer overfitting from training only on perfect electronic convolutions.

**Key insight for this project:** The Hermitian symmetry constraint is the right inductive bias for the Phase 2b/2c Fourier-plane mask. The current Phase 2 code learns an unconstrained complex phase mask; constraining it to Hermitian symmetry (a) halves parameters, (b) guarantees a real-valued spatial impulse response (a physically realizable convolution kernel), and (c) regularizes the Fourier landscape in a way the existing grid-size-dependent LR scaling is fighting blind.

---

## Quick-reference table — accuracy targets

| Phase | Architecture | Source paper | Target |
|---|---|---|---|
| 1a | Single-layer free-space | (current code, ad hoc) | 80–85% |
| 1b | Multi-layer free-space (3 layers) | Lin 2018, Sheng/Nisar 2024 | ≥90% |
| 2a | 4f + amax + curriculum | (current code, failure baseline) | poor shift generalization |
| 2b | 4f + motion pooling + Hermitian sym | Sadeghzadeh 2021, Schultz 2021 | ≥95% centered, robust to ±5 px |
| 2c | Tiled 4f + multi-layer D²NN | Sensors 2023 ODCNN | ≥97% |
