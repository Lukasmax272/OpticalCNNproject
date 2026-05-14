import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def smooth(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    # Colors for variants (matching benchmark.py)
    colors = {
        "Phase 1a": "#3b82f6", # Blue
        "Phase 1b": "#10b981", # Green
        "Phase 2b": "#f97316", # Orange
        "Phase 2c": "#facc15", # Yellow
    }

    # 1. Load Phase 1a (1 layer)
    if os.path.exists('history_1l_64p.npy'):
        data_1a = np.load('history_1l_64p.npy', allow_pickle=True).item()
        curve_1a = smooth(data_1a['accs'])
        ax.plot(np.arange(100, len(curve_1a)+100), curve_1a * 100, 
                label='Phase 1a: Single-layer', color=colors["Phase 1a"], linewidth=2)

    # 2. Load Phase 1b (3 layers)
    if os.path.exists('history_3l_64p.npy'):
        data_1b = np.load('history_3l_64p.npy', allow_pickle=True).item()
        curve_1b = smooth(data_1b['accs'])
        ax.plot(np.arange(100, len(curve_1b)+100), curve_1b * 100, 
                label='Phase 1b: 3-layer D²NN', color=colors["Phase 1b"], linewidth=2)

    # 3. Load Phase 2b (4f + motion pooling)
    if os.path.exists('history_phase2b_64p.npy'):
        data_2b = np.load('history_phase2b_64p.npy', allow_pickle=True).item()
        curve_2b = smooth(data_2b['accs'])
        ax.plot(np.arange(100, len(curve_2b)+100), curve_2b * 100, 
                label='Phase 2b: 4f + motion pooling', color=colors["Phase 2b"], linewidth=2)

    # 4. Load Phase 2c (ODCNN Hybrid)
    if os.path.exists('results/odcnn_hybrid.pt'):
        ckpt = torch.load('results/odcnn_hybrid.pt', map_location='cpu')
        curve_2c = smooth(ckpt['accs'])
        ax.plot(np.arange(100, len(curve_2c)+100), curve_2c * 100, 
                label='Phase 2c: ODCNN Hybrid (Shift±7)', color=colors["Phase 2c"], linewidth=2)

    ax.set_ylabel('Training Accuracy (%)', fontsize=12, color='#f8f8f2')
    ax.set_xlabel('Iteration (Batch Size = 64)', fontsize=12, color='#f8f8f2')
    ax.set_title('Training Convergence of Optical Architectures', fontsize=14, pad=15, color='#f8f8f2', fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True, facecolor='#282a36', edgecolor='none', labelcolor='#f8f8f2', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.2, color='#f8f8f2')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#f8f8f2')
    ax.spines['bottom'].set_color('#f8f8f2')
    
    plt.tight_layout()
    output_path = 'training_curves_all.png'
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
