"""Phase 2c: Faithful ODCNN reproduction (Sensors 2023).

Combines a tiled multi-kernel 4f convolution front-end with a 5-layer
complex-modulation D²NN classifier. Architecture per:

    "Optical convolutional neural network combining 4f system and D²NN",
    Sensors 2023.

Three documented divergences from project conventions
=====================================================

Divergence 1 — Amplitude encoding (not phase encoding)
    E_in = pixel_value + 0j  instead of  exp(i·π·pixel).
    Phase encoding gives uniform |E|² = 1 and a DC-dominated spectrum,
    so the 4f convolution kernels have no meaningful spatial structure to
    operate on.  Amplitude encoding matches the paper's Figure 1.

Divergence 2 — Complex modulation (not phase-only)
    Each D²NN layer has two learnable parameters per pixel:
        amplitude_logits → sigmoid → α ∈ (0, 1)
        phase_masks      → unbounded φ
        t = α · exp(j·φ)
    This matches the paper's eq. 3.  Phase mask stays unbounded per
    project convention; the sigmoid on amplitude is the minimal addition.
    Initializing logits at 0 gives α = 0.5 (mid-range, no gradient
    saturation).

Divergence 3 — MSE on softmax (not NSCE)
    loss = MSE(softmax(detector_bins), one_hot(label))
    per the paper's eq. 5.  Implemented in main_phase2_hybrid.py.

Stages (MNIST target — no ReLU)
================================
1.  Tiled multi-kernel 4f: 16 learnable 9×9 kernels in a 4×4 grid
    → FFT(spatial_layout) gives the Fourier-plane mask
    → E_conv = IFFT(FFT(E_in) · mask)
2.  (Skipped for MNIST — paper shows 97.32% linear vs 97.31% ReLU)
3.  5-layer complex-modulation D²NN classifier, 200×200 each
4.  Detector: 10 regions on 200×200 output, 5×2 grid of ~30×30 px bins
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TiledMultiKernelMask(nn.Module):
    """4f optical correlator mask producing spatially tiled feature maps.

    PSF(x, y) = Σ_m Σ_n Kernel_{(m-1)G+n}(x, y) · δ(x - Δx_m, y - Δy_n)
    (Sensors 2023 eq. 1).

    Kernels are parameterized in the spatial domain, tiled into a grid,
    then FFT'd to produce the Fourier-plane mask. The 50-pixel tile
    spacing on a 200×200 plane gives each feature map ~50×50 of space,
    preventing crosstalk for 28×28 inputs convolved with 9×9 kernels.
    """
    def __init__(self, num_kernels=16, kernel_size=9, grid_size=200):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.grid_size = grid_size

        self.grid_dim = int(math.sqrt(num_kernels))
        assert self.grid_dim ** 2 == num_kernels, "num_kernels must be a perfect square"
        self.cell_size = grid_size // self.grid_dim

        # 16 independently learnable real-valued spatial kernels
        self.spatial_kernels = nn.Parameter(
            torch.randn(num_kernels, kernel_size, kernel_size) * 0.1
        )

    def get_fourier_mask(self) -> torch.Tensor:
        """Build Fourier mask as FFT of the tiled spatial PSF layout.

        No normalization — the convolution result's magnitude is
        determined by the kernel values, which the optimizer controls.
        """
        layout = torch.zeros(
            self.grid_size, self.grid_size,
            dtype=torch.complex64, device=self.spatial_kernels.device
        )
        half = self.kernel_size // 2

        for i in range(self.grid_dim):
            for j in range(self.grid_dim):
                idx = i * self.grid_dim + j
                # Tile centers: 25, 75, 125, 175 for grid_dim=4, cell_size=50
                cy = 25 + i * 50 if self.grid_size == 200 else int((i + 0.5) * self.cell_size)
                cx = 25 + j * 50 if self.grid_size == 200 else int((j + 0.5) * self.cell_size)

                layout[cy - half:cy + half + 1, cx - half:cx + half + 1] = \
                    self.spatial_kernels[idx].to(torch.complex64)

        return torch.fft.fft2(layout)


class ComplexModulationD2NN(nn.Module):
    """Multi-layer D²NN with complex (amplitude + phase) modulation.

    Per Sensors 2023 eq. 3:  t_ℓ = α_ℓ · exp(j · φ_ℓ)

    Uses the same angular-spectrum propagation physics as MultiLayerD2NN
    in physics_sim.py (carrier subtraction, fftshift wrapping, DC-centred
    transfer function), but adds a learnable amplitude per pixel.

    This is a separate class from MultiLayerD2NN to avoid modifying the
    Phase 1 code path.  The physics is identical; only the modulation
    parameterization differs.
    """
    def __init__(
        self,
        num_layers: int = 5,
        grid_size: int = 200,
        z_total: float = 0.3,
        wavelength: float = 520e-9,
        pixel_size: float = 8e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.size = grid_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size

        # Equal spacing between layers and to detector
        self.z_spacing = z_total / (num_layers + 1)

        # Learnable parameters: amplitude logits + phase masks
        # Amplitude logits initialized at 0 → sigmoid(0) = 0.5
        # Phase masks initialized at 0 (Lin et al.)
        self.amplitude_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(grid_size, grid_size))
            for _ in range(num_layers)
        ])
        self.phase_masks = nn.ParameterList([
            nn.Parameter(torch.zeros(grid_size, grid_size))
            for _ in range(num_layers)
        ])

        # Print diffraction diagnostics
        D = grid_size * pixel_size
        spot = wavelength * z_total / D
        spot_px = spot / pixel_size
        print(f"ComplexModulationD2NN: {num_layers} layers, {grid_size}×{grid_size}")
        print(f"  z_spacing = {self.z_spacing*100:.2f} cm")
        print(f"  Diffraction spot ~= {spot_px:.1f} px")

        # Precompute transfer function (DC-centred, carrier subtracted)
        self._build_transfer_function()

    def _build_transfer_function(self) -> None:
        """Precompute angular-spectrum H for the uniform layer spacing.

        Carrier-phase subtraction: (kz - k) * z keeps the argument small
        enough for float32 cos/sin precision.  The subtracted phase is
        global and vanishes in |E|².
        """
        N = self.size
        f = (torch.arange(N, dtype=torch.float32) - N // 2) / (N * self.pixel_size)
        FX, FY = torch.meshgrid(f, f, indexing="ij")

        inv_lam = 1.0 / self.wavelength
        kz_arg = inv_lam ** 2 - FX ** 2 - FY ** 2
        propagating = (kz_arg >= 0).to(torch.float32)

        kz = 2 * math.pi * torch.sqrt(torch.clamp(kz_arg, min=0.0))
        k = 2 * math.pi * inv_lam

        H = torch.exp(1j * (kz - k) * self.z_spacing) * propagating
        self.register_buffer("H", H.to(torch.complex64))

    def _propagate(self, E: torch.Tensor) -> torch.Tensor:
        """Angular-spectrum propagation by one spacing."""
        spectrum = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
        spectrum = spectrum * self.H
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Multi-layer propagation with complex modulation.

        Args:
            input_field: complex64, shape (B, N, N).

        Returns:
            intensity: float32, shape (B, N, N).
        """
        E = input_field

        for i in range(self.num_layers):
            # Propagate to layer i
            E_pre = self._propagate(E)

            # Complex modulation: t = sigmoid(amp_logit) * exp(j * phi)
            amp = torch.sigmoid(self.amplitude_logits[i])
            phi = self.phase_masks[i]
            t = amp * torch.exp(1j * phi.to(torch.float32))
            E = E_pre * t

        # Propagate to detector
        E_out = self._propagate(E)

        # Square-law detection: |E|² without sqrt (Sadeghzadeh eq. 18)
        return E_out.real.square() + E_out.imag.square()


class ODCNNHybrid(nn.Module):
    """Full Phase 2c pipeline: Tiled 4f → 5-layer D²NN → Detector.

    See module docstring for the three documented divergences from
    project conventions.
    """
    def __init__(
        self,
        grid_size=200,
        num_kernels=16,
        kernel_size=9,
        d2nn_layers=5,
        d2nn_z_total=0.3,
        pixel_size=8e-6,
        motion_pool_sigma=0.0,
    ):
        super().__init__()
        self.grid_size = grid_size

        # Stage 1: Tiled multi-kernel 4f convolution
        self.tiled_mask = TiledMultiKernelMask(
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            grid_size=grid_size,
        )

        # Stage 3: Complex-modulation D²NN classifier
        self.d2nn = ComplexModulationD2NN(
            num_layers=d2nn_layers,
            grid_size=grid_size,
            z_total=d2nn_z_total,
            pixel_size=pixel_size,
        )

        # Optional motion pooling (project extension, not paper)
        if motion_pool_sigma > 0:
            ksize = max(3, int(6 * motion_pool_sigma) | 1)  # odd, ≥ 3σ each side
            coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
            g = torch.exp(-coords ** 2 / (2 * motion_pool_sigma ** 2))
            g = g / g.sum()
            kernel = (g[:, None] * g[None, :]).reshape(1, 1, ksize, ksize)
            self.register_buffer("motion_pool_kernel", kernel)
            self.use_motion_pool = True
        else:
            self.use_motion_pool = False

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ODCNN architecture.

        Args:
            input_field: complex64, shape (B, N, N).
                         Must be amplitude-encoded (E = pixel + 0j).

        Returns:
            intensity: float32, shape (B, N, N).
        """
        # --- Stage 1: Tiled multi-kernel 4f convolution ---
        # Lens 1: spatial → frequency (origin at (0,0) — no fftshift needed
        # since the mask is also built via fft2 in the same layout)
        spectrum = torch.fft.fft2(input_field)

        # Multiply by Fourier mask
        H = self.tiled_mask.get_fourier_mask()
        modulated = spectrum * H

        # Lens 2: frequency → spatial
        E_conv = torch.fft.ifft2(modulated)

        # --- Stage 2: Threshold ReLU — SKIPPED for MNIST ---
        # Paper shows 97.32% (linear) vs 97.31% (ReLU) on MNIST.
        # ReLU mostly helps Fashion-MNIST.

        # --- Stage 3: 5-layer D²NN classifier ---
        # Takes the complex field containing 16 spatially-tiled feature maps
        intensity = self.d2nn(E_conv)

        # --- Optional: motion pooling (project extension) ---
        if self.use_motion_pool:
            intensity = F.conv2d(
                intensity.unsqueeze(1),
                self.motion_pool_kernel,
                padding='same'
            ).squeeze(1)

        return intensity


def build_detector_masks_200(grid_size: int, device) -> torch.Tensor:
    """Build 10 detector region masks on a 200×200 output plane.

    Layout: 5×2 grid (5 columns, 2 rows).
        Row 1 (y≈66):  classes 0-4 at x = 33, 66, 100, 133, 166
        Row 2 (y≈133): classes 5-9 at x = 33, 66, 100, 133, 166

    Each region is ~30×30 px, well-spaced to avoid overlap.
    For non-200 grids, coordinates scale proportionally.

    Returns: (10, grid_size, grid_size) float32 mask tensor.
    """
    masks = torch.zeros(10, grid_size, grid_size, dtype=torch.float32, device=device)

    # Region half-width: ~15 px for 200×200 → 30×30 regions
    half = max(1, grid_size * 15 // 200)

    # Row y-centers at 1/3 and 2/3 of grid
    row_ys = [grid_size // 3, 2 * grid_size // 3]
    # Column x-centers: 5 evenly spaced
    col_xs = [grid_size * (2 * c + 1) // 10 for c in range(5)]

    for cls in range(10):
        row = cls // 5
        col = cls % 5
        cy = row_ys[row]
        cx = col_xs[col]
        y0 = max(0, cy - half)
        y1 = min(grid_size, cy + half)
        x0 = max(0, cx - half)
        x1 = min(grid_size, cx + half)
        masks[cls, y0:y1, x0:x1] = 1.0

    return masks
