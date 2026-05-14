import torch

x = torch.randn(64, 33, dtype=torch.complex64)
y = torch.fft.irfft2(x, s=(64, 64))
z = torch.fft.fft2(y)
print("x[0, 1]:", x[0, 1])
print("z[0, 1]:", z[0, 1])
