import time
import torch
import torch.nn.functional as F
from utils import load_mnist_generator
from physics_sim import OpticalSimulator, MultiLayerD2NN

def profile_run():
    DEVICE = torch.device("cpu")
    GRID = 64
    DATA_BATCH = 64
    TOTAL_ITERATIONS = 50

    # Match Phase 1 setup
    z_total = 0.096
    sim = MultiLayerD2NN(
        num_layers=1,
        grid_size=GRID,
        layer_spacings=[z_total / 2, z_total / 2],
        device=DEVICE
    )
    
    # max_shift=0 or 5, let's use 5 since that's what phase 2 uses
    gen = load_mnist_generator(batch_size=DATA_BATCH, device=DEVICE, train=True, max_shift=5)

    optimizer = torch.optim.Adam(sim.parameters(), lr=0.01)

    # dummy masks
    masks = torch.zeros(10, GRID, GRID, device=DEVICE)
    masks[:, 30:34, 30:34] = 1.0

    # warmup
    for _ in range(5):
        field, labels = next(gen)
        out = sim(field)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    data_time = 0.0
    compute_time = 0.0

    for _ in range(TOTAL_ITERATIONS):
        t0 = time.perf_counter()
        field, labels = next(gen)
        t1 = time.perf_counter()
        
        intensity = sim(field)
        bins = torch.einsum("bhw,khw->bk", intensity, masks)
        normalized = bins / bins.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        loss = F.cross_entropy(normalized, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t2 = time.perf_counter()
        
        data_time += (t1 - t0)
        compute_time += (t2 - t1)

    print(f"Total Data Time: {data_time:.4f}s")
    print(f"Total Compute Time: {compute_time:.4f}s")
    print(f"Data loading is {data_time / (data_time + compute_time) * 100:.1f}% of total time.")

if __name__ == '__main__':
    profile_run()
