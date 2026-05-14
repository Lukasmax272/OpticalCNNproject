"""Optics engine — Phase 2: 4f correlator (Fourier-plane filtering).

The forward path is:
    input field  →  FFT (Lens 1)  →  SLM (frequency domain)  →  IFFT (Lens 2)  →  intensity

The learnable phase mask operates in the Fourier plane, so a spatial shift
in the input field produces the same spatial shift in the output intensity.
This translation equivariance is the key structural advantage over Phase 1's
angular-spectrum propagation.

Inherits _apply_slm from OpticalSimulator so Phase 4's Kerr override
still works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from physics_sim import OpticalSimulator


class FourFSimulator(OpticalSimulator):
    """4f correlator: FFT → frequency-domain SLM → IFFT → intensity.

    No free-space transfer function H — the two ideal lenses handle
    propagation.  The SLM modulates the spatial frequency spectrum.
    """

    def __init__(self, size: int = 64, device=None):
        # Intentionally skip parent __init__ to avoid building H,
        # but keep the interface compatible.
        self.size = size
        self.device = device if device is not None else torch.device("cpu")
        # No H — the 4f system uses ideal lenses, not angular-spectrum propagation.

    def propagate(self, input_field: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        """FFT → SLM (Fourier plane) → IFFT → intensity.

        Args:
            input_field: complex64, shape (B, N, N).
            phase_mask:  real,      shape (..., N, N), broadcastable across batch.

        Returns:
            intensity:   float32,   shape (B, N, N).
        """
        # Lens 1: spatial → frequency domain.  DC-centred so the mask
        # layout matches physical intuition (low freqs in the middle).
        spectrum = torch.fft.fftshift(torch.fft.fft2(input_field), dim=(-2, -1))

        # SLM in the Fourier plane — same exp(i·φ) modulation as Phase 1,
        # but applied to the spectrum instead of the spatial field.
        modulated = self._apply_slm(spectrum, phase_mask)

        # Lens 2: back to spatial domain.
        out = torch.fft.ifft2(torch.fft.ifftshift(modulated, dim=(-2, -1)))

        # |E|² without the sqrt that .abs() would introduce.
        return out.real.square() + out.imag.square()


class MotionPoolingFourierMask(nn.Module):
    """4f optical correlator with motion pooling and Hermitian-symmetric mask.

    Motion pooling per Sadeghzadeh et al., IEEE Access 2021, eq. 17.
    Hermitian-symmetric mask per Schultz et al., IEEE Photonics Benelux 2021.
    """

    def __init__(self, size: int = 64, sigma: float = 3.0, kernel_size: int = 19, device=None):
        super().__init__()
        self.size = size
        self.device = device if device is not None else torch.device("cpu")

        # We parameterize the positive half of the Fourier kernel using an unbounded complex parameter.
        # This prevents vanishing gradients that occur with sigmoid-bounded amplitude.
        # Scale drift is prevented via weight decay in the optimizer.
        self.half_real = nn.Parameter(torch.empty(size, size // 2 + 1, device=self.device))
        self.half_imag = nn.Parameter(torch.empty(size, size // 2 + 1, device=self.device))

        self.init_binary_phase()

        # Motion pool kernel: 2D Gaussian
        def make_gaussian_kernel(sigma_val, ksize):
            coords = torch.arange(ksize, device=self.device) - (ksize - 1) / 2
            # Avoid div by zero if sigma_val is 0 (for ablation testing)
            if sigma_val <= 1e-5:
                g = torch.zeros(ksize, device=self.device)
                g[ksize // 2] = 1.0
            else:
                g = torch.exp(-coords**2 / (2 * sigma_val**2))
                g = g / g.sum()
            return (g[:, None] * g[None, :]).reshape(1, 1, ksize, ksize)

        self.register_buffer("motion_pool_kernel", make_gaussian_kernel(sigma, kernel_size))

    def init_binary_phase(self):
        """Initialize to binary phases (0 or pi)."""
        with torch.no_grad():
            self.half_real.copy_(torch.randint(0, 2, self.half_real.shape, device=self.device).float() * 2.0 - 1.0)
            self.half_imag.fill_(0.0)

    def full_fourier_mask(self) -> torch.Tensor:
        """Construct a full, perfectly Hermitian-symmetric complex mask."""
        half_complex = self.half_real + 1j * self.half_imag
        spatial_kernel = torch.fft.irfft2(half_complex, s=(self.size, self.size))
        full_mask = torch.fft.fft2(spatial_kernel)
        
        # Normalize the mask to act as a passive optical element (|H| <= 1).
        # We normalize the MAX amplitude to 1.0.
        full_mask = full_mask / full_mask.abs().amax().clamp(min=1e-8)
        
        # DC is at origin (0, 0), matching the output of fft2.
        return full_mask

    def propagate(self, input_field: torch.Tensor, phase_mask: torch.Tensor = None) -> torch.Tensor:
        """FFT -> apply symmetric mask -> IFFT -> square-law -> motion pool."""
        # Lens 1 (DC is at origin, no fftshift needed)
        spectrum = torch.fft.fft2(input_field)

        # Apply Hermitian-symmetric Fourier mask (no separate phase parameter)
        H = self.full_fourier_mask() if phase_mask is None else phase_mask
        modulated = spectrum * H

        # Lens 2
        out = torch.fft.ifft2(modulated)

        # Intensity (square-law detection)
        intensity = out.real.square() + out.imag.square()

        # Motion pooling (Gaussian blur)
        # Intensity has shape (B, N, N) -> reshape for conv2d
        pooled = F.conv2d(intensity.unsqueeze(1), self.motion_pool_kernel, padding='same').squeeze(1)

        return pooled
