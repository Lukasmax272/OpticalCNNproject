import matplotlib.pyplot as plt
import re

def parse_log(filename):
    iters = []
    accs = []
    losses = []
    with open(filename, 'r', encoding='utf-16', errors='ignore') as f:
        # Or typical ascii, but some logs might have weird encodings
        for line in f:
            if line.startswith('iter'):
                # Format: iter  1000  loss=1.7343  acc(last 100)=0.866
                match = re.search(r'iter\s+(\d+)\s+loss=([0-9.]+)\s+acc\(last \d+\)=([0-9.]+)', line)
                if match:
                    iters.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    accs.append(float(match.group(3)))
    return iters, losses, accs

def main():
    iters_base, loss_base, acc_base = parse_log('baseline_log.txt')
    iters_relu, loss_relu, acc_relu = parse_log('relu_log.txt')

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#1e1e1e')
    ax1.set_facecolor('#1e1e1e')
    ax2.set_facecolor('#1e1e1e')

    # Accuracy Plot
    ax1.plot(iters_base, acc_base, label='3-Layer Baseline', color='#50fa7b', linewidth=2)
    ax1.plot(iters_relu, acc_relu, label='3-Layer with Intensity ReLU (α=0.4)', color='#ff79c6', linewidth=2)
    ax1.set_title("Training Accuracy (last 100 iters)", color='#f8f8f2', fontsize=16, pad=10)
    ax1.set_xlabel("Iteration", color='#f8f8f2', fontsize=14)
    ax1.set_ylabel("Accuracy", color='#f8f8f2', fontsize=14)
    ax1.tick_params(colors='#f8f8f2')
    ax1.grid(True, alpha=0.2, color='#f8f8f2')
    ax1.legend(facecolor='#282a36', edgecolor='#44475a', labelcolor='#f8f8f2', fontsize=12)

    # Loss Plot
    ax2.plot(iters_base, loss_base, label='3-Layer Baseline', color='#50fa7b', linewidth=2)
    ax2.plot(iters_relu, loss_relu, label='3-Layer with Intensity ReLU (α=0.4)', color='#ff79c6', linewidth=2)
    ax2.set_title("Training Loss", color='#f8f8f2', fontsize=16, pad=10)
    ax2.set_xlabel("Iteration", color='#f8f8f2', fontsize=14)
    ax2.set_ylabel("Cross-Entropy Loss", color='#f8f8f2', fontsize=14)
    ax2.tick_params(colors='#f8f8f2')
    ax2.grid(True, alpha=0.2, color='#f8f8f2')
    ax2.legend(facecolor='#282a36', edgecolor='#44475a', labelcolor='#f8f8f2', fontsize=12)

    fig.suptitle("Phase 1c: Impact of Intensity Threshold ReLU", fontsize=20, color='#f8f8f2', fontweight='bold', y=1.05)
    plt.tight_layout()
    
    plot_path = "relu_comparison.png"
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved {plot_path}")

if __name__ == "__main__":
    main()
