"""MNIST data pipeline.

Pixel intensities are mapped to phase delays in [0, π] and the optical
input field is built as a unit-amplitude phase modulation:

        E_in[m, h, w] = exp(i · π · pixel[m, h, w])

This matches a phase-only SLM imprinting the digit onto a uniform beam.
The 28×28 digit is upsampled (if grid_size > 64) and zero-padded to fit
the simulator grid, preserving the ~44% fill ratio across grid sizes.

Phase 2c uses amplitude encoding instead (amplitude_encode=True):
        E_in[m, h, w] = pixel[m, h, w] + 0j
This gives |E|² that varies spatially, which is required for the 4f
convolution front-end to operate on spatial structure (Sensors 2023, Fig. 1).
"""

import torch
import torch.nn.functional as F
from torchvision import datasets


def load_mnist_generator(batch_size, device, train=True, data_root="./data",
                         max_shift=0, grid_size=64, upsample=True,
                         amplitude_encode=False):
    """Yields (input_field, labels) batches forever.

    Fields are computed on-the-fly per batch to avoid storing the full
    dataset at high resolutions (60k × 256 × 256 complex64 = ~31 GB).

    Args:
        batch_size:       digits per batch.
        device:           torch device for the yielded tensors.
        train:            MNIST train split if True, test if False.
        max_shift:        max random translation in pixels (both axes).
        grid_size:        side length of the output field (default 64).
                          When > 64, digits are bilinear-upsampled to maintain
                          the same fill ratio (28/64 ≈ 44%).
        amplitude_encode: if True, E_in = pixel_value + 0j (Phase 2c).
                          if False, E_in = exp(i·π·pixel) (Phase 1/2a/2b).

    Yields:
        input_field: complex64, shape (batch_size, grid_size, grid_size).
        labels:      long,      shape (batch_size,).
    """
    ds = datasets.MNIST(root=data_root, train=train, download=True)

    raw_imgs = ds.data.float() / 255.0     # (M, 28, 28) ∈ [0, 1]
    raw_labels = ds.targets.long()
    M = raw_imgs.shape[0]

    # Upsample factor: digit fills the same fraction of the grid as 28/64.
    factor = max(1, grid_size // 64) if upsample else 1
    digit_size = 28 * factor
    pad = (grid_size - digit_size) // 2

    # Precompute all images in CPU RAM to avoid bottlenecking the GPU training loop
    if factor > 1:
        raw_imgs = F.interpolate(
            raw_imgs.unsqueeze(1), size=digit_size,
            mode="bilinear", align_corners=False
        ).squeeze(1)

    # Keep as float32 to save RAM; convert to complex on-the-fly in the loop
    imgs = F.pad(raw_imgs, (pad, pad, pad, pad))

    while True:
        idx = torch.randint(0, M, (batch_size,))
        batch_pixels = imgs[idx].to(device)
        batch_labels = raw_labels[idx].to(device)

        if amplitude_encode:
            # Phase 2c: amplitude = pixel value, phase = 0
            batch_fields = batch_pixels.to(torch.complex64)
        else:
            # Phase 1/2a/2b: unit amplitude, phase = π·pixel
            batch_phases = batch_pixels * torch.pi
            batch_fields = torch.complex(torch.cos(batch_phases), torch.sin(batch_phases))

        if max_shift > 0:
            # Fully vectorized batched random roll
            shifts_y = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=device)
            shifts_x = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=device)
            
            H_dim, W_dim = batch_fields.shape[-2:]
            idx_y = (torch.arange(H_dim, device=device).unsqueeze(0) - shifts_y.unsqueeze(1)) % H_dim
            idx_x = (torch.arange(W_dim, device=device).unsqueeze(0) - shifts_x.unsqueeze(1)) % W_dim
            batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            
            batch_fields = batch_fields[batch_idx, idx_y.unsqueeze(2), idx_x.unsqueeze(1)]

        yield batch_fields, batch_labels