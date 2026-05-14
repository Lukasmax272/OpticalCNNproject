import torch

size = 64
half_amp_param = torch.randn(size, size // 2 + 1, requires_grad=True)
half_phase = torch.randn(size, size // 2 + 1, requires_grad=True)

half_complex = torch.sigmoid(half_amp_param) * torch.exp(1j * half_phase)
spatial_kernel = torch.fft.irfft2(half_complex, s=(size, size))
full_mask = torch.fft.fft2(spatial_kernel)

print("Max amplitude of full mask:", full_mask.abs().max().item())
