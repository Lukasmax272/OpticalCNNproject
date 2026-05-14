import torch
from four_f_sim import MotionPoolingFourierMask
from utils import load_mnist_generator
from main import build_bin_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid = 64

sim = MotionPoolingFourierMask(size=grid, sigma=3.0, kernel_size=19, device=DEVICE)
# We don't have the weights saved yet because the script hasn't finished!
# Wait, I cannot load the model weights because they are not saved!
