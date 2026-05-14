import torch
import numpy as np
import matplotlib.pyplot as plt
from physics_sim import MultiLayerD2NN
from main import build_bin_masks, GRID, DEVICE, TOTAL_ITERATIONS, DATA_BATCH, LEARNING_RATE
from utils import load_mnist_generator
import torch.nn.functional as F

def train_and_get_curve(num_layers):
    print(f"Training {num_layers} layers...")
    z_total = 0.096
    spacings = [z_total / (num_layers + 1)] * (num_layers + 1)
    sim = MultiLayerD2NN(
        num_layers=num_layers,
        grid_size=GRID,
        layer_spacings=spacings,
        device=DEVICE
    )
    
    gen = load_mnist_generator(batch_size=DATA_BATCH, device=DEVICE, train=True)
    bin_masks = build_bin_masks(GRID, DEVICE)
    optimizer = torch.optim.Adam(sim.parameters(), lr=LEARNING_RATE)

    accs = []
    
    for it in range(TOTAL_ITERATIONS):
        field, labels = next(gen)

        intensity = sim(field)
        bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)

        normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        loss = F.cross_entropy(normalized, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (bins.argmax(-1) == labels).float().mean().item()
            accs.append(acc)

    smoothed = np.convolve(accs, np.ones(100)/100, mode='valid')
    return smoothed

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    curve_1 = train_and_get_curve(1)
    curve_2 = train_and_get_curve(2)
    curve_3 = train_and_get_curve(3)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    x = np.arange(100, TOTAL_ITERATIONS + 1)
    
    colors = ['#ff79c6', '#8be9fd', '#50fa7b']
    
    ax.plot(x, curve_1 * 100, label='1 Layer', color=colors[0], linewidth=2, alpha=0.9)
    ax.plot(x, curve_2 * 100, label='2 Layers', color=colors[1], linewidth=2, alpha=0.9)
    ax.plot(x, curve_3 * 100, label='3 Layers', color=colors[2], linewidth=2, alpha=0.9)
    
    ax.set_ylabel('Training Accuracy (%)', fontsize=12, color='#f8f8f2')
    ax.set_xlabel('Iteration (Batch Size = 64)', fontsize=12, color='#f8f8f2')
    ax.set_title('Training Curves (100-batch rolling average)', fontsize=14, pad=15, color='#f8f8f2', fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True, facecolor='#282a36', edgecolor='none', labelcolor='#f8f8f2')
    ax.set_ylim(50, 100)
    ax.grid(True, linestyle='--', alpha=0.3, color='#f8f8f2')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#f8f8f2')
    ax.spines['bottom'].set_color('#f8f8f2')
    
    plt.tight_layout()
    plot_path = 'training_curves.png'
    plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved to {plot_path}")
