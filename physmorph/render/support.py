"""Target-free opacity validity for material-bound render primitives.

Physics particles remain authoritative.  These weights only say whether one of those
particles still has enough local support to represent a continuum Gaussian.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smoothstep(x, lo: float, hi: float):
    t = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class MaterialSupport:
    """Frozen source-material neighbours for render-only support checks."""

    neighbor: np.ndarray
    rest_length: np.ndarray

    @classmethod
    def from_rest(cls, rest_x: np.ndarray, k: int = 8) -> "MaterialSupport":
        from scipy.spatial import cKDTree

        rest = np.ascontiguousarray(rest_x, np.float32)
        if rest.ndim != 2 or rest.shape[1] != 3 or len(rest) < 2:
            raise ValueError("rest_x must have shape (N,3) with N >= 2")
        degree = min(max(int(k), 1), len(rest) - 1)
        dist, idx = cKDTree(rest).query(rest, k=degree + 1, workers=-1)
        return cls(np.ascontiguousarray(idx[:, 1:], np.int64),
                   np.ascontiguousarray(dist[:, 1:], np.float32))

    def opacity(self, x: np.ndarray, isolation_on: float = 4.0,
                isolation_off: float = 5.0, stretch_on: float = 1.75,
                stretch_off: float = 2.0, support_full: float = 2.0) -> np.ndarray:
        """Return ``[0,1]`` opacity multipliers without consulting the target.

        A primitive fades only when it is both an extreme current-body NN outlier and
        fewer than ``support_full`` frozen material bonds remain below the permitted
        stretch ramp.  The result is intentionally non-differentiable and must
        not become an optimisation reward for tearing material apart.
        """
        from scipy.spatial import cKDTree

        cur = np.ascontiguousarray(x, np.float32)
        n, degree = self.neighbor.shape
        if cur.shape != (n, 3):
            raise ValueError("x must have shape (N,3) matching the material graph")
        if isolation_off <= isolation_on:
            raise ValueError("isolation_off must be greater than isolation_on")
        if stretch_off <= stretch_on:
            raise ValueError("stretch_off must be greater than stretch_on")
        keep = min(max(float(support_full), 1.0), float(degree))
        dist = cKDTree(cur).query(cur, k=2, workers=-1)[0]
        nn_ratio = dist[:, 1] / max(float(np.median(dist[:, 1])), 1e-12)
        edge = cur[self.neighbor] - cur[:, None, :]
        stretch = np.linalg.norm(edge, axis=2) / np.maximum(self.rest_length, 1e-8)
        retained = (1.0 - _smoothstep(stretch, float(stretch_on),
                                      float(stretch_off))).sum(1)
        isolated = _smoothstep(nn_ratio, float(isolation_on), float(isolation_off))
        unsupported = 1.0 - _smoothstep(retained, 0.0, keep)
        return np.ascontiguousarray(1.0 - isolated * unsupported, np.float32)


# CUDA-render helpers; historical MaterialSupport API above remains unchanged.

import math
import torch
import torch.nn.functional as F


def live_support(distances, radius, spacing, *, k=8, smooth=False):
    """Self-first kNN distances; compact transition width is one target native spacing.

    The smooth contribution is <= the old hard indicator for every neighbor.
    No history, enlarged radius, opacity normalization, or missing-neighbor padding.
    """
    if k < 1 or distances.ndim != 2 or distances.shape[1] <= k:
        raise ValueError('support requires self plus k neighbor distances')
    if not all(math.isfinite(v) and v > 0 for v in (radius, spacing)):
        raise ValueError('support radius and target native spacing must be finite and positive')
    neighbors = distances[:, 1:k + 1]
    if smooth:
        q = ((radius - neighbors) / spacing).clamp(0, 1)
        contribution = q.square() * (3 - 2*q)
    else:
        contribution = (neighbors <= radius).to(distances.dtype)
    return (contribution.sum(1) / (0.5*k)).clamp(0, 1)


def normal_filter_size(height, scaled=False):
    """Old 3px footprint, or nearest odd radius scaled from radius1 at height1080."""
    if height < 1:
        raise ValueError('image height must be positive')
    return 2*max(1, int(math.floor(height/1080 + .5))) + 1 if scaled else 3


def filter_normal_buffer(normal_buffer, coverage, kernel_size=3):
    """Filter premultiplied normals only; the caller composites ORIGINAL coverage."""
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError('normal filter footprint must be a positive odd integer')
    def smooth(image):
        return F.avg_pool2d(image.permute(2, 0, 1)[None], kernel_size, 1,
                            kernel_size//2)[0].permute(1, 2, 0)
    return F.normalize(2*smooth(normal_buffer) / smooth(coverage[..., None]).clamp_min(1e-3)-1,
                       dim=-1, eps=1e-6)


def rest_affine_inverse(offsets, rank_ratio=1e-4):
    """No ridge: unsupported 3D rest neighborhoods explicitly remain refit-only."""
    gram = offsets.double().transpose(1, 2) @ offsets.double()
    decomposed = [torch.linalg.eigh(chunk) for chunk in gram.split(8192)]
    values = torch.cat([pair[0] for pair in decomposed])
    vectors = torch.cat([pair[1] for pair in decomposed])
    valid = (values[:, 0] > rank_ratio*values[:, 2]) & (values[:, 2] > 0)
    safe = torch.where(valid[:, None], values, torch.ones_like(values))
    inverse = ((vectors/safe[:, None]) @ vectors.transpose(1, 2)).to(offsets.dtype)
    return inverse, valid


def transport_affine_normal(rest, current, inverse, normal, residual_limit=.5, rank_ratio=1e-4):
    """Full 3D affine fit and oriented cofactor normal; never align sign to history."""
    affine = (current.transpose(1, 2) @ rest) @ inverse
    a, b, c = affine.unbind(2)
    cofactor = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a),
                            torch.linalg.cross(a, b)), 2)
    determinant = (a*cofactor[:, :, 0]).sum(1)
    scale2 = affine.square().sum((1, 2))/3
    residual = (current-rest @ affine.transpose(1, 2)).norm(dim=(1, 2)) / current.norm(dim=(1, 2)).clamp_min(1e-20)
    carried = (cofactor @ normal[..., None])[..., 0]
    length = carried.norm(dim=1)
    relative_det = determinant/scale2.pow(1.5).clamp_min(1e-20)
    valid = (torch.isfinite(carried).all(1) & torch.isfinite(residual)
             & (relative_det > rank_ratio) & (length > rank_ratio*scale2)
             & (residual <= residual_limit))
    return carried/length.clamp_min(1e-20)[:, None], valid


class MaterialShadingNormals:
    """Fixed-ID material affine frames affect shading only; invalid fits use CURRENT refit."""
    def __init__(self, count, device, degree=32):
        self.degree = degree
        self.anchored = torch.zeros(count, dtype=torch.bool, device=device)
        self.neighbors = torch.zeros((count, degree), dtype=torch.long, device=device)
        self.offsets = torch.zeros((count, degree, 3), device=device)
        self.inverse = torch.zeros((count, 3, 3), device=device)
        self.normal = torch.zeros((count, 3), device=device)

    def update(self, x, refit, neighbors, exposed, locked):
        result = refit.clone()
        attempted = self.anchored & ~locked
        accepted = torch.zeros_like(attempted)
        failed = torch.zeros_like(attempted)
        ids = attempted.nonzero().squeeze(1)
        if len(ids):
            current = x[self.neighbors[ids]]-x[ids, None]
            carried, valid = transport_affine_normal(self.offsets[ids], current,
                                                      self.inverse[ids], self.normal[ids])
            result[ids[valid]] = carried[valid]
            accepted[ids[valid]] = True
            failed[ids[~valid]] = True
            self.anchored[ids[~valid]] = False
        new = (exposed & ~self.anchored & ~locked).nonzero().squeeze(1)
        reanchored = torch.zeros_like(attempted)
        rank_failed = torch.zeros_like(attempted)
        if len(new):
            graph = neighbors[new, 1:self.degree+1]
            rest = x[graph]-x[new, None]
            inverse, valid = rest_affine_inverse(rest)
            selected = new[valid]
            self.neighbors[selected] = graph[valid]
            self.offsets[selected] = rest[valid]
            self.inverse[selected] = inverse[valid]
            self.normal[selected] = refit[selected]
            self.anchored[selected] = True
            reanchored[selected] = True
            rank_failed[new[~valid]] = True
        return result, dict(attempted=attempted, transported=accepted, invalid_fit=failed,
                            current_refit=~accepted & ~locked, new_anchor=reanchored,
                            rest_rank_rejected=rank_failed, pinned_frozen=locked.clone())
