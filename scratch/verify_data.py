import numpy as np
import json
import base64

def verify_js_data():
    with open("browser/data.js", "r") as f:
        lines = f.readlines()
    
    meta = json.loads(lines[0].split("const META = ")[1].rstrip(";\n"))
    grid = meta["grid_size"]
    steps = meta["z_steps"]
    
    p1_b64 = lines[2].split("const P1_DATA_B64 = '")[1].rstrip("';\n")
    p1_bytes = base64.b64decode(p1_b64)
    p1_data = np.frombuffer(p1_bytes, dtype=np.uint8).reshape(10, steps, grid, grid)
    
    p2b_b64 = lines[3].split("const P2B_DATA_B64 = '")[1].rstrip("';\n")
    p2b_bytes = base64.b64decode(p2b_b64)
    p2b_data = np.frombuffer(p2b_bytes, dtype=np.uint8).reshape(10, steps, grid, grid)

    # Build bin masks
    def build_bin_masks(N):
        layout = [
            (0, 1/6, 1/4), (1, 1/6, 2/4), (2, 1/6, 3/4),
            (3, 1/2, 1/5), (4, 1/2, 2/5), (5, 1/2, 3/5), (6, 1/2, 4/5),
            (7, 5/6, 1/4), (8, 5/6, 2/4), (9, 5/6, 3/4),
        ]
        half = N // 16
        masks = np.zeros((10, N, N))
        for cls, ry, rx in layout:
            cy, cx = int(ry * N), int(rx * N)
            y0, y1 = max(0, cy - half), min(N, cy + half)
            x0, x1 = max(0, cx - half), min(N, cx + half)
            masks[cls, y0:y1, x0:x1] = 1.0
        return masks

    masks = build_bin_masks(grid)
    
    print("Verification of Detector Plane (Final Step):")
    for d in range(10):
        intensity = p1_data[d, steps-1].astype(float)
        bins = [np.sum(intensity * masks[k]) for k in range(10)]
        pred = np.argmax(bins)
        print(f"  Digit {d}: Predicted {pred}, OK? {pred==d}")

    print("\nPhase 2b Verification:")
    for d in range(10):
        intensity = p2b_data[d, steps-1].astype(float)
        bins = [np.sum(intensity * masks[k]) for k in range(10)]
        pred = np.argmax(bins)
        print(f"  Digit {d}: Predicted {pred}, OK? {pred==d}")

if __name__ == "__main__":
    verify_js_data()
