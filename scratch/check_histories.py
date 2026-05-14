import numpy as np
import os

histories = {
    "Phase 1a (1l)": "history_1l_64p.npy",
    "Phase 1b (3l)": "history_3l_64p.npy",
    "Phase 2b (4f)": "history_phase2b_64p.npy"
}

for name, path in histories.items():
    if os.path.exists(path):
        h = np.load(path, allow_pickle=True).item()
        accs = h.get("accs", [])
        if accs:
            final_acc = sum(accs[-100:]) / len(accs[-100:])
            print(f"{name} final training accuracy: {final_acc:.4f}")
        else:
            print(f"{name} has no accuracy data in history.")
    else:
        print(f"{name} history file not found.")
