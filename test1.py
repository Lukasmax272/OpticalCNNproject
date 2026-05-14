import torch
device = torch.device('cpu')
param = torch.nn.Parameter(torch.ones(2, 2, requires_grad=True))
psf = torch.zeros(4, 4, device=device)
psf[1:3, 1:3] = param
loss = psf.sum()
loss.backward()
print("Grad:", param.grad)