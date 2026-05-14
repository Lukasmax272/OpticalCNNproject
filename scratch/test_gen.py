import torch
from utils import load_mnist_generator

print("Testing generator...")
gen = load_mnist_generator(64, torch.device("cpu"), max_shift=5, grid_size=64)
fields, labels = next(gen)
print("Fields shape:", fields.shape)
print("Labels shape:", labels.shape)
