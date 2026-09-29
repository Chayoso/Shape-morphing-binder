"""Stateless surface primitives shared by the P302 loss and archive renderer.

KNN identities, support indicators and normal-donor selection are discrete active
sets, refreshed at EVERY forward evaluation. Selected distances and continuous
density/normal/covariance paths remain differentiable. No appearance parameters
or mutable pin latch are optimized. This representation has no rest guarantee.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .knn_gpu import knn_self_torch
from .studio import DensityNormals
from .support import live_support


def pack_covariance(covariance):
    return torch.stack([covariance[:, i, j] for i, j in
                        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], -1).contiguous()


def tangent_covariance(normals, sigma):
    """Unit-normal tangent-isotropic splat, without eigenvector derivatives."""
    eye = torch.eye(3, device=normals.device, dtype=normals.dtype)
    return sigma[:, None, None].square() * (eye - (15. / 16.) *
                                            normals[:, :, None] * normals[:, None, :])


@dataclass
class SurfacePrimitives:
    normals: torch.Tensor
    covariance: torch.Tensor
    opacity: torch.Tensor
    sigma: torch.Tensor
    support: torch.Tensor


class SurfaceGaussians:
    """Reference calibration is fixed; candidate geometry is always live."""
    def __init__(self, center, radius, spacing, coverage_radius, *, cuda_only=True):
        if cuda_only and not center.is_cuda:
            raise ValueError('Surface GS production requires CUDA')
        if not all(math.isfinite(v) and v > 0 for v in (radius, spacing, coverage_radius)):
            raise ValueError('Surface GS calibration must be finite and positive')
        self.center = center.detach().clone()
        self.radius, self.spacing, self.coverage_radius = radius, spacing, coverage_radius
        self.normals = DensityNormals(self.center, radius, spacing)
        self.cuda_only = cuda_only

    @classmethod
    def from_reference(cls, reference, *, cuda_only=True):
        with torch.no_grad():
            distances, _ = knn_self_torch(reference, 9)
            center = reference.mean(0)
            return cls(center, float((reference-center).norm(dim=1).max()),
                       float(distances[:, 1].median()), float(distances[:, 8].median()),
                       cuda_only=cuda_only)

    def __call__(self, x):
        if len(x) < 33 or x.device != self.center.device:
            raise ValueError('Surface GS needs >=33 particles on its reference device')
        if self.cuda_only and not x.is_cuda:
            raise ValueError('Surface GS production requires CUDA')
        # Warp's large-cloud KNN distances are detached; use only its identities.
        with torch.no_grad():
            _, neighbors = knn_self_torch(x.detach(), 33)
        distances = torch.linalg.vector_norm(x[neighbors] - x[:, None, :], dim=-1)
        support = live_support(distances, self.coverage_radius, self.spacing)
        sigma = self.spacing * (distances[:, 8] / self.coverage_radius).clamp(1., 4.)
        normals, magnitude = self.normals(x)
        strong = magnitude >= torch.quantile(magnitude[::max(1, len(x)//100000)], .6)
        nearest = neighbors[:, 1:33]
        strong_neighbors = strong[nearest]
        chosen = nearest[torch.arange(len(x), device=x.device), strong_neighbors.float().argmax(1)]
        normals = torch.where((~strong & strong_neighbors.any(1))[:, None], normals[chosen], normals)
        for _ in range(2):
            normals = F.normalize(normals[neighbors].mean(1), dim=1, eps=1e-9)
        # A zero density normal has no surface direction. Explicit radial fallback
        # keeps positive covariance; center-coincident rows use a fixed z normal.
        radial = x-self.center
        fallback = F.normalize(radial, dim=1, eps=1e-9)
        fallback = torch.where((radial.norm(dim=1) > 1e-9)[:, None], fallback,
                               x.new_tensor((0., 0., 1.)))
        normals = torch.where((normals.norm(dim=1) > 1e-8)[:, None], normals, fallback)
        covariance = tangent_covariance(normals, sigma)
        return SurfacePrimitives(normals, covariance, .92*support, sigma, support)

    def metadata(self):
        return dict(model='surface_gs_stateless_v1', radius=self.radius, spacing=self.spacing,
                    coverage_radius=self.coverage_radius, normal_blur_sp=3., sigma_range_sp=[1., 4.],
                    opacity_scale=.92, normal_axis_ratio=.25,
                    gradient='live continuous attributes; discrete KNN/support/donors refreshed per forward',
                    limitations='Hard-zero support has no GS recovery gradient; no learned opacity; no pin appearance latch')
