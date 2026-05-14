import torch
import time

B = 128
H, W = 64, 64
x = torch.randn(B, H, W, device='cpu')
shifts_y = torch.randint(-5, 6, (B,), device='cpu')
shifts_x = torch.randint(-5, 6, (B,), device='cpu')

def batched_roll(x, shifts_y, shifts_x):
    B, H, W = x.shape
    device = x.device
    idx_y = (torch.arange(H, device=device).unsqueeze(0) - shifts_y.unsqueeze(1)) % H
    idx_x = (torch.arange(W, device=device).unsqueeze(0) - shifts_x.unsqueeze(1)) % W
    batch_idx = torch.arange(B, device=device).view(B, 1, 1)
    return x[batch_idx, idx_y.unsqueeze(2), idx_x.unsqueeze(1)]

t0 = time.time()
res1 = batched_roll(x, shifts_y, shifts_x)
print("Batched roll time:", time.time() - t0)

t0 = time.time()
res2 = x.clone()
for i in range(B):
    res2[i] = torch.roll(res2[i], shifts=(shifts_y[i].item(), shifts_x[i].item()), dims=(-2, -1))
print("For loop time:", time.time() - t0)

print("Match:", torch.allclose(res1, res2))
