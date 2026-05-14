import torch
import torch.nn.functional as F
import torch.nn as nn

grid = 64
batch = 16

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.half_real = nn.Parameter(torch.randn(grid, grid // 2 + 1))
        self.half_imag = nn.Parameter(torch.randn(grid, grid // 2 + 1))

    def forward(self, x):
        half_complex = torch.complex(self.half_real, self.half_imag)
        spatial_kernel = torch.fft.irfft2(half_complex, s=(grid, grid))
        H = torch.fft.fft2(spatial_kernel)
        
        # What happens to H scale over steps?
        return H

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = torch.randn(batch, grid, grid, dtype=torch.complex64)
bin_masks = torch.rand(10, grid, grid)

for step in range(100):
    H = model(x)
    spectrum = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    modulated = spectrum * torch.fft.fftshift(H, dim=(-2, -1))
    out = torch.fft.ifft2(torch.fft.ifftshift(modulated, dim=(-2, -1)))
    intensity = out.real.square() + out.imag.square()
    
    bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
    normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    loss = F.cross_entropy(normalized, torch.randint(0, 10, (batch,)))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}, Loss {loss.item():.4f}, H mean abs: {H.abs().mean().item():.4f}, grad mean: {model.half_real.grad.abs().mean().item():.4e}")
