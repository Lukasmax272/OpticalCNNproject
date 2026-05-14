import torch
import torch.nn.functional as F
from torchvision import datasets
import time

def test_gen():
    t0 = time.time()
    ds = datasets.MNIST(root="./data", train=True, download=True)
    raw_imgs = ds.data.float() / 255.0
    labels = ds.targets.long()
    
    grid_size = 64
    factor = max(1, grid_size // 64)
    digit_size = 28 * factor
    pad = (grid_size - digit_size) // 2

    if factor > 1:
        raw_imgs = F.interpolate(raw_imgs.unsqueeze(1), size=digit_size, mode="bilinear", align_corners=False).squeeze(1)
        
    imgs = F.pad(raw_imgs, (pad, pad, pad, pad))
    phases = imgs * torch.pi
    fields = torch.complex(torch.cos(phases), torch.sin(phases))
    print("Precompute time:", time.time() - t0)
    print("Fields shape:", fields.shape)

test_gen()
