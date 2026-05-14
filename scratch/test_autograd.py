import torch

grid = 64
x1 = torch.randn(grid, grid//2+1, dtype=torch.complex64, requires_grad=True)
H1 = torch.fft.fft2(torch.fft.irfft2(x1, s=(grid, grid)))
loss1 = H1.abs().sum()
loss1.backward()
print("Default norm grad mean:", x1.grad.abs().mean().item())

x2 = torch.randn(grid, grid//2+1, dtype=torch.complex64, requires_grad=True)
H2 = torch.fft.fft2(torch.fft.irfft2(x2, s=(grid, grid), norm='ortho'), norm='ortho')
loss2 = H2.abs().sum()
loss2.backward()
print("Ortho norm grad mean:", x2.grad.abs().mean().item())
