import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess

from physics_sim import MultiLayerD2NN
from main import build_bin_masks, GRID, DEVICE
from utils import load_mnist_generator

def evaluate_model(num_layers, phase_mask_np):
    z_total = 0.096
    spacings = [z_total / (num_layers + 1)] * (num_layers + 1)
    sim = MultiLayerD2NN(
        num_layers=num_layers,
        grid_size=GRID,
        layer_spacings=spacings,
        device=DEVICE
    )
    
    for i in range(num_layers):
        with torch.no_grad():
            sim.phase_masks[i].copy_(torch.from_numpy(phase_mask_np[i]).to(DEVICE))
            
    bin_masks = build_bin_masks(GRID, DEVICE)
    gen = load_mnist_generator(batch_size=256, device=DEVICE, train=False)
    
    correct_per_class = np.zeros(10)
    total_per_class = np.zeros(10)
    
    # Evaluate 10,240 samples
    for _ in range(40):
        field, labels = next(gen)
        with torch.no_grad():
            intensity = sim(field)
            bins = torch.einsum("bhw,khw->bk", intensity, bin_masks)
            preds = bins.argmax(-1)
            
            for c in range(10):
                mask = (labels == c)
                correct_per_class[c] += (preds[mask] == c).sum().item()
                total_per_class[c] += mask.sum().item()
                
    return correct_per_class / np.maximum(total_per_class, 1)

def train_and_eval(num_layers):
    print(f"--- Training {num_layers}-layer model ---")
    subprocess.run([".venv\\Scripts\\python", "main.py", "--num-layers", str(num_layers), "--seed"], check=True)
    phase_mask_np = np.load("learned_phase_mask.npy")
    accs = evaluate_model(num_layers, phase_mask_np)
    print(f"Overall Acc: {accs.mean()*100:.1f}%\n")
    return accs

if __name__ == "__main__":
    accs_1 = train_and_eval(1)
    accs_2 = train_and_eval(2)
    accs_3 = train_and_eval(3)
    
    classes = np.arange(10)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    colors = ['#ff79c6', '#8be9fd', '#50fa7b']
    
    rects1 = ax.bar(classes - width, accs_1 * 100, width, label='1 Layer', color=colors[0], edgecolor='none')
    rects2 = ax.bar(classes, accs_2 * 100, width, label='2 Layers', color=colors[1], edgecolor='none')
    rects3 = ax.bar(classes + width, accs_3 * 100, width, label='3 Layers', color=colors[2], edgecolor='none')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, color='#f8f8f2')
    ax.set_xlabel('MNIST Class', fontsize=12, color='#f8f8f2')
    ax.set_title('Per-Class Accuracy by D²NN Depth', fontsize=14, pad=15, color='#f8f8f2', fontweight='bold')
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, fontsize=11, color='#f8f8f2')
    ax.tick_params(axis='y', colors='#f8f8f2')
    
    ax.legend(loc='lower right', frameon=True, facecolor='#282a36', edgecolor='none', labelcolor='#f8f8f2')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#f8f8f2')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#f8f8f2')
    ax.spines['bottom'].set_color('#f8f8f2')
    
    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#f8f8f2', alpha=0.7)
    
    plt.tight_layout()
    plot_path = 'layer_accuracy_chart.png'
    plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved bar chart to {plot_path}")
