"""Optics engine — Phase 1: angular-spectrum free-space propagation.

The forward path is:
    input field  →  SLM (phase mask)  →  free-space propagation  →  intensity

`_apply_slm` and the (fft → H → ifft) block in `propagate` are the seams
that future phases swap:
    Phase 2 replaces the propagation block with a 4f correlator (FFT, mask,
        IFFT applied with the SLM in the frequency plane).
    Phase 4 overrides `_apply_slm` for the Kerr nonlinearity:
        E_out = E_in · exp(i · (φ + γ|E_in|²)).
"""

import math
import torch


class OpticalSimulator:

    def __init__(
        self,
        size: int = 64,
        pixel_size: float = 8e-6,    # 8 µm
        wavelength: float = 520e-9,  # 520 nm
        z: float = 0.096,            # 9.6 cm
        device=None,
    ):
        self.size = size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.z = z
        self.device = device if device is not None else torch.device("cpu")

        self._build_transfer_function()

    # ------------------------------------------------------------- precompute
    def _build_transfer_function(self) -> None:
        """Precompute H(fx, fy), DC-centred to match the fftshift-wrapped FFT.

        The on-axis carrier phase k·z is subtracted before building H — at
        λ=520 nm, z=9.6 cm the un-subtracted phase exceeds 10⁶ rad, which
        kills float32 precision inside cos/sin.  It's a global phase that
        vanishes in |E|², so the physics is unchanged.
        """
        N = self.size
        # DC-centred frequency coords: [-N/2, …, N/2 − 1] / (N · dx)
        f = (torch.arange(N, dtype=torch.float32) - N // 2) / (N * self.pixel_size)
        FX, FY = torch.meshgrid(f, f, indexing="ij")

        inv_lam = 1.0 / self.wavelength
        kz_arg = inv_lam ** 2 - FX ** 2 - FY ** 2
        propagating = (kz_arg >= 0).to(torch.float32)

        kz = 2 * math.pi * torch.sqrt(torch.clamp(kz_arg, min=0.0))
        k = 2 * math.pi * inv_lam
        H = torch.exp(1j * (kz - k) * self.z) * propagating

        self.H = H.to(torch.complex64).to(self.device)

    # --------------------------------------------------------------- SLM hook
    def _apply_slm(self, field: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        """E_out = E_in · exp(i·φ).  Phase 4 overrides for the Kerr term."""
        return field * torch.exp(1j * phase_mask.to(torch.float32))

    # ---------------------------------------------------------------- forward
    def propagate(self, input_field: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        """SLM → angular-spectrum propagation → intensity.

        Args:
            input_field: complex64, shape (B, N, N).
            phase_mask:  real,      shape (..., N, N), broadcastable across batch.

        Returns:
            intensity:   float32,   shape (B, N, N).
        """
        modulated = self._apply_slm(input_field, phase_mask)

        # FFT, centre DC, multiply by DC-centred H, un-centre, IFFT.
        spectrum = torch.fft.fftshift(torch.fft.fft2(modulated), dim=(-2, -1))
        spectrum = spectrum * self.H
        out = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))

        # |E|² without the sqrt that .abs() would introduce.
        return out.real.square() + out.imag.square()


class MultiLayerD2NN(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        grid_size: int,
        layer_spacings: list[float],
        wavelength: float = 520e-9,
        pixel_size: float = 8e-6,
        device=None,
        intensity_relu_alpha: float | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.size = grid_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.device = device if device is not None else torch.device("cpu")
        self.layer_spacings = layer_spacings
        self.intensity_relu_alpha = intensity_relu_alpha
        
        # Verify layer spacings length
        if len(layer_spacings) != num_layers + 1:
            raise ValueError(f"Expected {num_layers + 1} layer spacings, got {len(layer_spacings)}")

        # Print diffraction spot vs bin size
        total_z = sum(layer_spacings)
        D = grid_size * pixel_size
        diffraction_spot = wavelength * total_z / D
        half_base = grid_size // 16
        bin_size = 2 * half_base * pixel_size
        
        print(f"MultiLayerD2NN built. N={num_layers} layers.")
        print(f"  Diffraction spot: {diffraction_spot*1e6:.1f} µm")
        print(f"  Detector bin size: {bin_size*1e6:.1f} µm")
        if bin_size < 2 * diffraction_spot:
            print(f"  Warning: detector bin size ({bin_size*1e6:.1f} µm) is less than 2x diffraction spot ({diffraction_spot*1e6*2:.1f} µm).")

        self._build_transfer_functions()
        
        # Zero-initialized phase masks (Lin et al.)
        self.phase_masks = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(grid_size, grid_size, device=self.device))
            for _ in range(num_layers)
        ])

    def _build_transfer_functions(self) -> None:
        """Precompute H for each unique spacing to save memory and compute."""
        self.H_dict = {}
        unique_z = set(self.layer_spacings)
        
        N = self.size
        f = (torch.arange(N, dtype=torch.float32) - N // 2) / (N * self.pixel_size)
        FX, FY = torch.meshgrid(f, f, indexing="ij")
        
        inv_lam = 1.0 / self.wavelength
        kz_arg = inv_lam ** 2 - FX ** 2 - FY ** 2
        propagating = (kz_arg >= 0).to(torch.float32)
        
        kz = 2 * math.pi * torch.sqrt(torch.clamp(kz_arg, min=0.0))
        k = 2 * math.pi * inv_lam
        
        for z in unique_z:
            H = torch.exp(1j * (kz - k) * z) * propagating
            self.H_dict[z] = H.to(torch.complex64).to(self.device)

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Multi-layer angular-spectrum propagation.
        
        Args:
            input_field: complex64, shape (B, N, N).
            
        Returns:
            intensity: float32, shape (B, N, N).
        """
        E = input_field
        
        for i in range(self.num_layers):
            # Propagate to layer i
            z = self.layer_spacings[i]
            spectrum = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
            spectrum = spectrum * self.H_dict[z]
            E_pre = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))
            
            # Optional Intensity Threshold ReLU between layers.
            # Models a camera-readout-and-re-encoding step, which destroys the input phase.
            # Physically realizable via sCMOS curve adjustment or photorefractive crystals
            # per Sensors 2023 ODCNN paper eq. 7 and Section 4.
            if i > 0 and self.intensity_relu_alpha is not None:
                I = E_pre.real.square() + E_pre.imag.square()
                I_max = I.amax(dim=(-2, -1), keepdim=True)
                I_relu = torch.where(I >= self.intensity_relu_alpha * I_max, I, torch.zeros_like(I))
                E_pre = torch.sqrt(I_relu).to(torch.complex64)
            
            # Phase modulation
            E = E_pre * torch.exp(1j * self.phase_masks[i].to(torch.float32))
            
        # Propagate to detector
        z_out = self.layer_spacings[-1]
        spectrum = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
        spectrum = spectrum * self.H_dict[z_out]
        E_out = torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))
        
        return E_out.real.square() + E_out.imag.square()