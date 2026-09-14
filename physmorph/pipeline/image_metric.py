"""Spatially nonlocal residuals of actual rendered images; no state displacement."""
import torch


class ScreenedImageResidual:
    """(I - ell^2 Laplacian)^(-1/2) with mirrored image boundaries.

    ell is a fraction of image width/height, independent of pixel resolution.
    The DC component is preserved: changing foreground area remains penalized.
    This defines a different image norm, not optimal transport or a gradient gain.
    """
    def __init__(self, height, width, length=.08, device="cpu", dtype=torch.float32):
        if length <= 0:
            raise ValueError("screen length must be positive")
        ky = 2*torch.pi*torch.fft.fftfreq(2*height, d=1/height, device=device, dtype=dtype)
        kx = 2*torch.pi*torch.fft.rfftfreq(2*width, d=1/width, device=device, dtype=dtype)
        self.weight = (1+length**2*(ky[:, None]**2+kx[None, :]**2)).rsqrt()
        self.height, self.width = height, width

    def __call__(self, difference):
        if difference.shape[-2:] != (self.height, self.width):
            raise ValueError("image shape changed")
        mirrored = torch.cat([difference, difference.flip(-1)], -1)
        mirrored = torch.cat([mirrored, mirrored.flip(-2)], -2)
        spectrum = torch.fft.rfft2(mirrored, norm="ortho")
        filtered = torch.fft.irfft2(spectrum*self.weight,
                                    s=(2*self.height, 2*self.width), norm="ortho")
        return filtered[..., :self.height, :self.width].contiguous()
