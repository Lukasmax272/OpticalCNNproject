import torch
import torch.nn.functional as F
from four_f_hybrid import ODCNNHybrid

sim = ODCNNHybrid(grid_size=200, num_kernels=16, kernel_size=9, d2nn_layers=2)
x = torch.randn(2, 200, 200, dtype=torch.complex64)
# Stage 1
spectrum = torch.fft.fft2(x)
H = sim.tiled_mask.get_full_mask()
modulated = spectrum * H
E_conv = torch.fft.ifft2(modulated)

I_conv = E_conv.real.square() + E_conv.imag.square()
I_max = I_conv.amax(dim=(-2, -1), keepdim=True)
I_relu = torch.where(I_conv >= 0.4 * I_max, I_conv, torch.zeros_like(I_conv))
print("Non-zeros before ReLU:", (I_conv > 1e-5).sum().item())
print("Non-zeros after ReLU:", (I_relu > 0).sum().item())
