import torch
import torch.nn as nn

size = 64
half_real = nn.Parameter(torch.randn(size, size // 2 + 1))
half_imag = nn.Parameter(torch.randn(size, size // 2 + 1))

opt = torch.optim.Adam([half_real, half_imag], lr=0.01, weight_decay=1e-4)

for i in range(100):
    half_complex = half_real + 1j * half_imag
    spatial_kernel = torch.fft.irfft2(half_complex, s=(size, size))
    full_mask = torch.fft.fft2(spatial_kernel)
    
    # Scale invariant loss! We just multiply the mask by some data and then normalize.
    # To simulate scale invariance, we just try to make the mask match a target pattern,
    # but we normalize the mask first.
    target = torch.ones_like(full_mask).real
    
    norm_mask = full_mask / full_mask.abs().amax().clamp(min=1e-8)
    loss = torch.mean((norm_mask.real - target)**2)
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}, max abs: {full_mask.abs().amax().item():.4f}")
