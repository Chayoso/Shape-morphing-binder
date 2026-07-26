"""
Per-episode Gaussian covariance/opacity/rotation optimization.
Position is FROZEN — only appearance parameters are updated via render loss.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _pca_normals_torch(x, k=16):
    """Compute PCA normals from particle positions."""
    from scipy.spatial import cKDTree
    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    tree = cKDTree(x_np)
    _, idx = tree.query(x_np, k=k + 1)
    idx = idx[:, 1:]
    normals = np.zeros_like(x_np)
    for i in range(len(x_np)):
        neighbors = x_np[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal direction
    return normals


def init_rotation_from_normals(normals):
    """
    Convert normal vectors to quaternions that align the z-axis with the normal.
    Returns (N, 4) quaternions (w, x, y, z).
    """
    N = len(normals)
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # default identity

    for i in range(N):
        n = normals[i]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            continue
        n = n / n_norm
        dot = np.clip(np.dot(z_axis, n), -1.0, 1.0)
        if dot > 0.9999:
            quats[i] = [1, 0, 0, 0]
        elif dot < -0.9999:
            quats[i] = [0, 1, 0, 0]
        else:
            axis = np.cross(z_axis, n)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = np.arccos(dot)
            s = np.sin(angle / 2)
            quats[i] = [np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]
    return quats


class CovarianceOptimizer:
    """Manages per-particle covariance/opacity/rotation optimization across episodes."""

    def __init__(self, N, sigma0, opacity_val, device, cfg=None):
        cfg = cfg or {}
        self.N = N
        self.sigma0 = sigma0
        self.device = device
        self.iters_per_ep = int(cfg.get('cov_iters_per_ep', 5))
        self.lr = float(cfg.get('cov_lr', 0.003))
        self.scale_max_ratio = float(cfg.get('cov_scale_max_ratio', 1.5))
        self.scale_min_ratio = float(cfg.get('cov_scale_min_ratio', 0.3))
        self.scale_reg = float(cfg.get('cov_scale_reg', 0.01))
        self.optimize_rotation = bool(cfg.get('cov_optimize_rotation', True))
        self.reset_optimizer = bool(cfg.get('cov_reset_optimizer', False))

        log_scale_init = np.log(sigma0)
        self.log_scale_min = np.log(sigma0 * self.scale_min_ratio)
        self.log_scale_max = np.log(sigma0 * self.scale_max_ratio)

        self.log_scale = torch.full((N, 3), log_scale_init, device=device, requires_grad=True)
        self.logit_opacity = torch.full(
            (N,), float(np.log(opacity_val / (1.0 - opacity_val))),
            device=device, requires_grad=True
        )

        # Rotation: initialized later from PCA normals
        self.rotation = torch.zeros((N, 4), device=device)
        self.rotation[:, 0] = 1.0  # identity
        self.rotation.requires_grad = False  # set True after init_from_normals

        params = [self.log_scale, self.logit_opacity]
        if self.optimize_rotation:
            params.append(self.rotation)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        # Pre-computed DT maps (set once from targets)
        self.dt_maps = []
        self._initialized = False

    def init_from_positions(self, x_np):
        """Initialize rotation from PCA normals of particle positions."""
        if not self.optimize_rotation:
            return
        normals = _pca_normals_torch(x_np, k=16)
        quats = init_rotation_from_normals(normals)
        with torch.no_grad():
            self.rotation.copy_(torch.from_numpy(quats).float().to(self.device))
        self.rotation.requires_grad = True
        # Re-create optimizer with rotation
        self.optimizer = torch.optim.Adam(
            [self.log_scale, self.logit_opacity, self.rotation],
            lr=self.lr
        )

    def set_targets(self, targets):
        """Pre-compute DT maps from target alpha maps."""
        self.dt_maps = []
        for tgt in targets:
            tgt_np = tgt.detach().cpu().numpy() if torch.is_tensor(tgt) else np.array(tgt)
            tgt_bin = (tgt_np > 0.5).astype(np.float64)
            dt_out = distance_transform_edt(1.0 - tgt_bin)
            dt_in = distance_transform_edt(tgt_bin)
            signed_dt = (dt_out - dt_in).astype(np.float32)
            self.dt_maps.append(torch.from_numpy(signed_dt).to(self.device))
        self._initialized = True

    def step(self, x_np, renderers, targets, campos_list, color, rcfg=None,
             particle_mask=None):
        """
        Run covariance optimization for self.iters_per_ep iterations.
        Position x_np is FROZEN.
        Returns metrics dict.
        """
        if not self._initialized:
            self.set_targets(targets)

        # Reset optimizer momentum each episode (particle positions changed)
        if self.reset_optimizer:
            params = [self.log_scale, self.logit_opacity]
            if self.optimize_rotation:
                params.append(self.rotation)
            self.optimizer = torch.optim.Adam(params, lr=self.lr)

        x_t = torch.from_numpy(x_np).float().to(self.device)
        N = len(x_np)
        color_t = torch.tensor(color, device=self.device).float().unsqueeze(0).expand(N, -1)
        norm3d = torch.zeros((N, 3), device=self.device)
        norm3d[:, 2] = 1.0

        # Use subset of views per step to save VRAM
        view_indices = list(range(0, len(renderers), 2))

        total_alpha_mse = 0.0
        total_dt = 0.0

        for it in range(self.iters_per_ep):
            self.optimizer.zero_grad()

            for v in view_indices:
                rend = renderers[v]
                scale = torch.exp(self.log_scale)
                opt_opacity = torch.sigmoid(self.logit_opacity)

                # Normalize rotation quaternion
                if self.optimize_rotation:
                    rot = F.normalize(self.rotation, p=2, dim=1)
                else:
                    rot = self.rotation

                means2D = torch.zeros((N, 3), device=self.device, requires_grad=True)

                color_out, depth_out, norm_out, alpha_raw, radii, extra = rend.rasterizer(
                    means3D=x_t,
                    means2D=means2D,
                    opacities=opt_opacity.unsqueeze(-1),
                    colors_precomp=color_t,
                    scales=scale,
                    rotations=rot,
                    norm3Ds_precomp=norm3d,
                )

                alpha = alpha_raw[0] if alpha_raw.dim() == 3 else alpha_raw

                # DT loss
                dt_map = self.dt_maps[v]
                if dt_map.shape != alpha.shape:
                    dt_map = F.interpolate(dt_map[None, None], size=alpha.shape,
                                           mode='bilinear', align_corners=False)[0, 0]
                l_dt = (alpha * dt_map).mean()

                # IoU loss
                tgt = targets[v].to(self.device)
                if tgt.shape != alpha.shape:
                    tgt = F.interpolate(tgt[None, None], size=alpha.shape,
                                        mode='bilinear', align_corners=False)[0, 0]
                intersection = (alpha * tgt).sum()
                union = alpha.sum() + tgt.sum() - intersection
                l_iou = 1.0 - intersection / (union + 1e-7)

                view_loss = (l_dt + 0.5 * l_iou) / len(view_indices)

                # Scale regularization
                if v == view_indices[0]:
                    s_reg = self.scale_reg * ((self.log_scale - np.log(self.sigma0)) ** 2).mean()
                    view_loss = view_loss + s_reg / len(view_indices)

                view_loss.backward()

                if it == self.iters_per_ep - 1:
                    with torch.no_grad():
                        total_alpha_mse += float(((alpha - tgt) ** 2).mean().item())
                        total_dt += float(l_dt.item())

            # NaN guard before stepping
            valid_grads = True
            for p in [self.log_scale, self.logit_opacity]:
                if p.grad is not None and torch.isnan(p.grad).any():
                    valid_grads = False
                    p.grad.zero_()
            if self.optimize_rotation and self.rotation.grad is not None:
                if torch.isnan(self.rotation.grad).any():
                    valid_grads = False
                    self.rotation.grad.zero_()

            if valid_grads:
                self.optimizer.step()

            # Clamp scale & fix NaN
            with torch.no_grad():
                # Replace any NaN in parameters
                nan_mask = torch.isnan(self.log_scale)
                if nan_mask.any():
                    self.log_scale[nan_mask] = np.log(self.sigma0)
                self.log_scale.clamp_(self.log_scale_min, self.log_scale_max)

                nan_mask_o = torch.isnan(self.logit_opacity)
                if nan_mask_o.any():
                    self.logit_opacity[nan_mask_o] = 2.944  # sigmoid(2.944) ≈ 0.95

                if self.optimize_rotation:
                    nan_mask_r = torch.isnan(self.rotation).any(dim=1)
                    if nan_mask_r.any():
                        self.rotation.data[nan_mask_r] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
                    self.rotation.data = F.normalize(self.rotation.data, p=2, dim=1)

        n_views = max(len(view_indices), 1)
        scale_np = torch.exp(self.log_scale).detach().cpu().numpy()
        opa_np = torch.sigmoid(self.logit_opacity).detach().cpu().numpy()

        return {
            'cov_alpha_mse': total_alpha_mse / n_views,
            'cov_dt': total_dt / n_views,
            'cov_scale_mean': float(scale_np.mean()),
            'cov_scale_min': float(scale_np.min()),
            'cov_scale_max': float(scale_np.max()),
            'cov_opacity_mean': float(opa_np.mean()),
            'cov_opacity_min': float(opa_np.min()),
        }

    def get_render_params(self):
        """Return current optimized parameters for rendering."""
        with torch.no_grad():
            scale = torch.exp(self.log_scale)
            opacity = torch.sigmoid(self.logit_opacity)
            rot = F.normalize(self.rotation, p=2, dim=1) if self.optimize_rotation else self.rotation
        return {
            'scale': scale.cpu().numpy(),
            'opacity': opacity.cpu().numpy(),
            'rotation': rot.cpu().numpy(),
        }
