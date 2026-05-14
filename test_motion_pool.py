import torch
from four_f_sim import MotionPoolingFourierMask

def test_hermitian_symmetry():
    model = MotionPoolingFourierMask(size=64)
    mask = model.full_fourier_mask()
    
    # Check shape
    assert mask.shape == (64, 64)
    
    # Check Hermitian symmetry
    # For a mask with DC in the center (via fftshift), H(-k) = H*(k)
    # The flipped version of the mask should equal the conjugate.
    flipped = torch.roll(torch.flip(mask, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)).conj()
    if not torch.allclose(mask, flipped, atol=1e-6):
        print(f"Max diff: {torch.max(torch.abs(mask - flipped))}")
    assert torch.allclose(mask, flipped, atol=1e-6)

    # Check that spatial domain is purely real when propagated
    # ifft2 of mask should be purely real
    spatial_from_mask = torch.fft.ifft2(mask)
    assert torch.allclose(spatial_from_mask.imag, torch.zeros_like(spatial_from_mask.imag), atol=1e-5)
    
    print("Hermitian symmetry test passed!")

if __name__ == "__main__":
    test_hermitian_symmetry()
