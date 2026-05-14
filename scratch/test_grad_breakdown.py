import torch
import torch.nn.functional as F

grid = 64
batch = 16

half_real = torch.randn(grid, grid // 2 + 1, requires_grad=True)
half_imag = torch.randn(grid, grid // 2 + 1, requires_grad=True)

half_complex = torch.complex(half_real, half_imag)
spatial_kernel = torch.fft.irfft2(half_complex, s=(grid, grid))
H = torch.fft.fft2(spatial_kernel)
H.retain_grad()

x = torch.randn(batch, grid, grid, dtype=torch.complex64)
bin_masks = torch.rand(10, grid, grid)

spectrum = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
modulated = spectrum * torch.fft.fftshift(H, dim=(-2, -1))
out = torch.fft.ifft2(torch.fft.ifftshift(modulated, dim=(-2, -1)))
out.retain_grad()

intensity = out.real.square() + out.imag.square()
bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
bins.retain_grad()

normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
loss = F.cross_entropy(normalized, torch.randint(0, 10, (batch,)))

loss.backward()
print("dL / dNorm:", loss.item())
print("bins mean:", bins.mean().item())
print("dL / dBins mean abs:", bins.grad.abs().mean().item())
print("dL / dout mean abs:", out.grad.abs().mean().item())
print("dL / dH mean abs:", H.grad.abs().mean().item())
print("dL / dHalfReal mean abs:", half_real.grad.abs().mean().item())
