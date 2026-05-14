import torch
import math

size = 8

print("=== Option A: rfft2/irfft2 ===")
# Store the Fourier-plane mask as a real-valued field, backed by half-spectrum FFTs
half_spectrum = torch.randn(size, size // 2 + 1, dtype=torch.complex64)
# What if it's "Store the Fourier-plane mask as a real-valued field"?
# Let's parameterize a real-valued field:
spatial_kernel = torch.randn(size, size)
fourier_mask_A = torch.fft.fftshift(torch.fft.fft2(spatial_kernel))
print("Option A mask shape:", fourier_mask_A.shape)
# Test Hermitian symmetry: M(-k) = M*(k)
# Since it is fftshift-ed, we flip the dims
flipped_A = torch.flip(fourier_mask_A, dims=(-2, -1)).conj()
print("Option A matches flipped:", torch.allclose(fourier_mask_A, flipped_A, atol=1e-6))

print("\n=== Option B: Explicit symmetry reconstruction ===")
half_mask = torch.randn(size // 2, size, dtype=torch.complex64)
middle_row = torch.randn(1, size // 2, dtype=torch.complex64)
dc = torch.randn(1, 1, dtype=torch.float32) + 0j
nyquist = torch.randn(1, 1, dtype=torch.float32) + 0j

# Construct full mask
top = half_mask
mid = torch.cat([middle_row, dc, middle_row.flip(-1).conj(), nyquist], dim=-1) # wait, DC is at size//2
# Actually, explicitly reconstructing 2D Hermitian symmetry is very tedious.

# Let's try PyTorch's rfft2 trick for phase mask!
print("\n=== Option C: rfft2 trick for phase mask ===")
spatial_phase = torch.randn(size, size)
fourier_complex = torch.fft.fft2(spatial_phase)
# But wait, this makes fourier_complex Hermitian symmetric. If we want exp(i * phi) to be Hermitian symmetric,
# phi must be odd.
spatial_odd = torch.randn(size, size)
# make odd
spatial_odd = (spatial_odd - torch.flip(spatial_odd, dims=(0, 1))) / 2
print("Odd max error:", torch.max(torch.abs(spatial_odd + torch.flip(spatial_odd, dims=(0, 1)))))

# Phase initialized to 0 or pi
binary_phase = (torch.randint(0, 2, (size, size)).float() * math.pi)
binary_phase = (binary_phase - torch.flip(binary_phase, dims=(0, 1))) / 2
# wait, (pi - 0) / 2 = pi/2, which is not 0 or pi.

# The prompt says: "Store the Fourier-plane mask as a real-valued field (no separate phase parameter), backed by half-spectrum FFTs."
# If there is no separate phase parameter, then the mask IS the real-valued field!
# BUT a mask must be Hermitian symmetric to have a real spatial impulse response!
# If the mask is a real-valued field, and it is Hermitian symmetric, then it must be REAL and EVEN.
# Let's see if a real and even mask works:
real_even_spatial = torch.randn(size, size)
real_even_spatial = (real_even_spatial + torch.flip(real_even_spatial, dims=(0, 1))) / 2
fourier_mask_C = torch.fft.fftshift(torch.fft.fft2(real_even_spatial))
print("Fourier mask C is real:", torch.allclose(fourier_mask_C.imag, torch.zeros_like(fourier_mask_C.imag)))
print("Fourier mask C is Hermitian symmetric:", torch.allclose(fourier_mask_C, torch.flip(fourier_mask_C, dims=(0, 1)).conj()))

