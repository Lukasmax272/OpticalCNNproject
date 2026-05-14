import sys
sys.path.append(".")
import torch
from utils import load_mnist_generator

gen = load_mnist_generator(batch_size=1, device=torch.device('cpu'), train=False, grid_size=64)
field, labels = next(gen)
digit_img = torch.angle(field[0]).numpy()

print("digit_img shape:", digit_img.shape)
print("min:", digit_img.min())
print("max:", digit_img.max())
print("mean:", digit_img.mean())
print("value at 0,0:", digit_img[0,0])
print("value at center:", digit_img[32,32])
