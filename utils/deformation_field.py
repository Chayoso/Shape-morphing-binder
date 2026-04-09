"""
Coarse Grid Deformation Field — Manual Gradient Projection

Forward:  x' = x + W @ u      (trilinear interpolation)
Backward: g_u = W^T @ g_x     (weighted accumulation)

No torch autograd required. Works with numpy-based renderer.
"""

import numpy as np


class CoarseDeformationField:
    """
    3D grid of displacement vectors with explicit forward/backward.

    Forward: interpolate grid displacements to particle positions
    Backward: project particle-space gradients back to grid nodes

    Args:
        grid_res: resolution (e.g., 8 means 8x8x8 nodes)
        bbox_min: (3,) min corner
        bbox_max: (3,) max corner
    """

    def __init__(self, grid_res=8, bbox_min=None, bbox_max=None):
        self.grid_res = grid_res
        self.M = grid_res ** 3  # total grid nodes

        # Displacements: (M, 3) flattened grid
        self.u = np.zeros((self.M, 3), dtype=np.float32)

        # Bounding box
        self.bbox_min = np.array(bbox_min, dtype=np.float32) if bbox_min is not None else np.zeros(3, dtype=np.float32)
        self.bbox_max = np.array(bbox_max, dtype=np.float32) if bbox_max is not None else np.ones(3, dtype=np.float32)

        # Cached interpolation data (computed in precompute_weights)
        self.node_ids = None   # (N, 8) int
        self.weights = None    # (N, 8) float32
        self.active_mask = None  # (N,) bool, optional

    def set_bbox(self, positions, padding=2.0):
        """Set bounding box from particle positions with padding."""
        self.bbox_min = positions.min(axis=0) - padding
        self.bbox_max = positions.max(axis=0) + padding

    def reset(self):
        """Reset displacements to zero."""
        self.u[:] = 0.0

    def precompute_weights(self, x, active_mask=None):
        """
        Compute trilinear interpolation weights for particle positions.
        Call once per correction cycle (before inner optimization loop).

        Args:
            x: (N, 3) particle positions

        Stores:
            self.node_ids: (N, 8) indices into flattened grid
            self.weights: (N, 8) trilinear weights
        """
        N = x.shape[0]
        R = self.grid_res
        bbox_size = self.bbox_max - self.bbox_min
        bbox_size = np.maximum(bbox_size, 1e-6)

        # Normalize to [0, R-1]
        normalized = (x - self.bbox_min) / bbox_size  # [0, 1]
        grid_coords = normalized * (R - 1)
        grid_coords = np.clip(grid_coords, 0, R - 1 - 1e-6)

        # Integer and fractional parts
        ix0 = grid_coords[:, 0].astype(np.int32)
        iy0 = grid_coords[:, 1].astype(np.int32)
        iz0 = grid_coords[:, 2].astype(np.int32)

        fx = (grid_coords[:, 0] - ix0).astype(np.float32)
        fy = (grid_coords[:, 1] - iy0).astype(np.float32)
        fz = (grid_coords[:, 2] - iz0).astype(np.float32)

        ix1 = np.minimum(ix0 + 1, R - 1)
        iy1 = np.minimum(iy0 + 1, R - 1)
        iz1 = np.minimum(iz0 + 1, R - 1)

        # 8 corner node indices (flattened: idx = ix * R*R + iy * R + iz)
        def flat(ix, iy, iz):
            return ix * R * R + iy * R + iz

        self.node_ids = np.stack([
            flat(ix0, iy0, iz0),  # 000
            flat(ix0, iy0, iz1),  # 001
            flat(ix0, iy1, iz0),  # 010
            flat(ix0, iy1, iz1),  # 011
            flat(ix1, iy0, iz0),  # 100
            flat(ix1, iy0, iz1),  # 101
            flat(ix1, iy1, iz0),  # 110
            flat(ix1, iy1, iz1),  # 111
        ], axis=1).astype(np.int64)  # (N, 8)

        # 8 trilinear weights
        self.weights = np.stack([
            (1 - fx) * (1 - fy) * (1 - fz),  # 000
            (1 - fx) * (1 - fy) * fz,          # 001
            (1 - fx) * fy * (1 - fz),          # 010
            (1 - fx) * fy * fz,                  # 011
            fx * (1 - fy) * (1 - fz),          # 100
            fx * (1 - fy) * fz,                  # 101
            fx * fy * (1 - fz),                  # 110
            fx * fy * fz,                          # 111
        ], axis=1).astype(np.float32)  # (N, 8)

        if active_mask is None:
            self.active_mask = np.ones((N,), dtype=bool)
        else:
            self.active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if self.active_mask.shape[0] != N:
                raise ValueError(f"active_mask length mismatch: expected {N}, got {self.active_mask.shape[0]}")
            self.weights[~self.active_mask] = 0.0

    def forward(self, x):
        """
        Apply deformation field to particle positions.

        Args:
            x: (N, 3) particle positions

        Returns:
            x_prime: (N, 3) deformed positions
        """
        assert self.node_ids is not None, "Call precompute_weights first"
        u_local = self.u[self.node_ids]                          # (N, 8, 3)
        disp = (self.weights[..., None] * u_local).sum(axis=1)  # (N, 3)
        return x + disp

    def backward(self, grad_x):
        """
        Project particle-space gradients to grid-node gradients.
        g_u = W^T @ g_x

        Args:
            grad_x: (N, 3) dL/dx' from renderer

        Returns:
            grad_u: (M, 3) dL/du for grid nodes
        """
        assert self.node_ids is not None, "Call precompute_weights first"
        grad_u = np.zeros((self.M, 3), dtype=np.float32)
        for k in range(8):
            idx = self.node_ids[:, k]         # (N,)
            w = self.weights[:, k:k+1]        # (N, 1)
            contrib = w * grad_x              # (N, 3)
            np.add.at(grad_u, idx, contrib)
        return grad_u

    def field_smoothness_grad(self):
        """
        Gradient of field smoothness loss:
        L_smooth = Σ_{(j,k)∈E} ||u_j - u_k||²

        Returns:
            grad_u: (M, 3)
        """
        R = self.grid_res
        u_3d = self.u.reshape(R, R, R, 3)
        grad_3d = np.zeros_like(u_3d)

        # x-direction neighbors
        grad_3d[:-1] += 2 * (u_3d[:-1] - u_3d[1:])
        grad_3d[1:]  += 2 * (u_3d[1:] - u_3d[:-1])
        # y-direction neighbors
        grad_3d[:, :-1] += 2 * (u_3d[:, :-1] - u_3d[:, 1:])
        grad_3d[:, 1:]  += 2 * (u_3d[:, 1:] - u_3d[:, :-1])
        # z-direction neighbors
        grad_3d[:, :, :-1] += 2 * (u_3d[:, :, :-1] - u_3d[:, :, 1:])
        grad_3d[:, :, 1:]  += 2 * (u_3d[:, :, 1:] - u_3d[:, :, :-1])

        return grad_3d.reshape(self.M, 3) / self.M

    def field_smoothness_loss(self):
        """Compute field smoothness loss value."""
        R = self.grid_res
        u_3d = self.u.reshape(R, R, R, 3)
        loss = 0.0
        loss += ((u_3d[1:] - u_3d[:-1]) ** 2).sum()
        loss += ((u_3d[:, 1:] - u_3d[:, :-1]) ** 2).sum()
        loss += ((u_3d[:, :, 1:] - u_3d[:, :, :-1]) ** 2).sum()
        return float(loss / self.M)

    def clip_displacements(self, max_norm):
        """Clip displacement magnitudes (trust-region)."""
        norms = np.linalg.norm(self.u, axis=1, keepdims=True)
        scale = np.where(norms > max_norm, max_norm / (norms + 1e-8), 1.0)
        self.u *= scale

    def project_mirror_symmetry(self, axis=0, strength=1.0):
        """
        Project the field to a mirror-symmetric subspace.

        For a mirror across the chosen axis:
          u_axis(-x) = -u_axis(x)
          u_other(-x) =  u_other(x)

        Args:
            axis: 0, 1, or 2
            strength: 0 -> no-op, 1 -> hard projection
        """
        if strength <= 0:
            return
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"symmetry axis must be 0/1/2, got {axis}")

        R = self.grid_res
        u_grid = self.u.reshape(R, R, R, 3)
        partner = np.flip(u_grid, axis=axis).copy()
        partner[..., axis] *= -1.0
        projected = 0.5 * (u_grid + partner)
        self.u = ((1.0 - strength) * self.u + strength * projected.reshape(self.M, 3)).astype(np.float32)

    def max_displacement(self):
        """Maximum displacement magnitude."""
        return float(np.linalg.norm(self.u, axis=1).max())

    def active_fraction(self):
        """Fraction of particles affected by this field."""
        if self.active_mask is None or self.active_mask.size == 0:
            return 1.0
        return float(self.active_mask.mean())

    def update(self, grad_u, lr=0.01):
        """Simple gradient descent step."""
        self.u -= lr * grad_u
