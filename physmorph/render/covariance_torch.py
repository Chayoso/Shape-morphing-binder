"""Device-resident covariance decomposition for the splat renderer."""
from __future__ import annotations

import torch


def rotation_to_quaternion(matrix):
    """Proper rotation matrices to unit WXYZ quaternions, including half turns."""
    r = matrix
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    q2 = torch.stack((1 + r00 + r11 + r22, 1 + r00 - r11 - r22,
                      1 - r00 + r11 - r22, 1 - r00 - r11 + r22), -1).clamp_min(0)
    qabs = q2.sqrt()
    candidates = torch.stack((
        torch.stack((q2[..., 0], r[..., 2, 1] - r[..., 1, 2],
                     r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1),
        torch.stack((r[..., 2, 1] - r[..., 1, 2], q2[..., 1],
                     r[..., 1, 0] + r[..., 0, 1], r[..., 0, 2] + r[..., 2, 0]), -1),
        torch.stack((r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] + r[..., 0, 1],
                     q2[..., 2], r[..., 2, 1] + r[..., 1, 2]), -1),
        torch.stack((r[..., 1, 0] - r[..., 0, 1], r[..., 0, 2] + r[..., 2, 0],
                     r[..., 2, 1] + r[..., 1, 2], q2[..., 3]), -1),
    ), -2) / (2 * qabs.clamp_min(1e-12)[..., :, None])
    index = qabs.argmax(-1)[..., None, None].expand(*qabs.shape[:-1], 1, 4)
    q = candidates.gather(-2, index).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def decompose_cov_torch(cov, batch_size=8192):
    """Same float64 eigensolve and eigenvalue floor as the NumPy export path."""
    sym = 0.5 * (cov + cov.transpose(-1, -2))
    # cuSOLVER's batched workspace query rejects very large 3x3 batches on some
    # CUDA builds; bound each solve independently of the particle population.
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    flat = sym.reshape(-1, 3, 3)
    parts = [torch.linalg.eigh(chunk.double()) for chunk in flat.split(batch_size)]
    values = torch.cat([part[0] for part in parts]).reshape(*sym.shape[:-2], 3)
    axes = torch.cat([part[1] for part in parts]).reshape(*sym.shape[:-2], 3, 3)
    axes = axes.clone()
    axes[..., :, 0] *= torch.where(torch.linalg.det(axes) < 0, -1., 1.)[..., None]
    return values.clamp_min(1e-12).sqrt().float(), rotation_to_quaternion(axes).float()


def world_to_view_torch(camera, target):
    """Graphdeco +z-forward view; use the legacy camera's float64 arithmetic."""
    camera, target = camera.double(), target.double()
    forward = target - camera
    forward = forward / (forward.norm() + 1e-9)
    up = camera.new_tensor((0., 1., 0.))
    right = torch.linalg.cross(up, forward)
    right = right / (right.norm() + 1e-9)
    rotation = torch.stack((right, torch.linalg.cross(forward, right), forward))
    view = torch.eye(4, dtype=camera.dtype, device=camera.device)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ camera
    return view.float()
