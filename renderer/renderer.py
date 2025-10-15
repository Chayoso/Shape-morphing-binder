# renderer.py
# -----------------------------------------------------------------------------
# Robust wrapper around diff_gaussian_rasterization's GaussianRasterizer.
# - Accepts either the official Graphdeco build or forks that return multiple
#   G-buffer outputs (e.g. color, depth, normals, alpha, radii, ...).
# - Computes means2D on CPU using the provided projection matrix to guarantee
#   stable tiling and avoid "white screen" issues when means2D is missing.
# - Handles both covariance fast-path (Nx6 packed) and decomposition fallback
#   (Σ -> scales + rotations as quaternion [xyzw]).
# - Heuristically *parses* rasterizer outputs to pick depth/alpha even when
#   the tuple order differs across forks.
#
# NOTE: Code is NumPy/Torch-agnostic outside the rasterizer call. All public
#       returns are NumPy arrays for easy downstream use.
# -----------------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Any
import os, sys
import numpy as np

# ---- Import rasterizer (try several namespaces) ------------------------------
_import_ok = False
_last_err  = None
try:
    import torch
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    _import_ok = True
except Exception as e:
    _last_err = e
    # Try alternative module names or local submodules
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, ".."))
    candidates = [
        os.path.join(repo_root, "submodules", "diff-gaussian-rasterization"),
        os.path.join(repo_root, "diff-gaussian-rasterization"),
        os.path.join(repo_root, "gaussian-splatting"),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import torch  # re-import just in case
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        _import_ok = True
    except Exception as e2:
        _last_err = e2

if not _import_ok:
    raise RuntimeError(
        "diff_gaussian_rasterization is required. Install/build the official Graphdeco "
        "rasterizer or make sure your local submodule is importable."
    ) from _last_err

# ---- Small helpers -----------------------------------------------------------
def _to_torch(x, device="cuda", dtype=None):
    if dtype is None:
        import torch as _torch
        dtype = _torch.float32
    import torch as _torch
    t = _torch.as_tensor(x, dtype=dtype)
    return t.to(device) if device else t

def _as_cov6(cov: np.ndarray) -> np.ndarray:
    """Convert Nx3x3 symmetric covariance to Nx6 packed [xx, xy, xz, yy, yz, zz]."""
    if cov.ndim == 2 and cov.shape[1] == 6:
        return cov.astype(np.float32, copy=False)
    if cov.ndim != 3 or cov.shape[1:] != (3, 3):
        raise ValueError("cov must be (N,3,3) or (N,6).")
    xx = cov[:, 0, 0]; xy = cov[:, 0, 1]; xz = cov[:, 0, 2]
    yy = cov[:, 1, 1]; yz = cov[:, 1, 2]; zz = cov[:, 2, 2]
    return np.stack([xx, xy, xz, yy, yz, zz], axis=1).astype(np.float32)

def _quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Return XYZW quaternion from 3x3 rotation matrix."""
    m = R; t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def _scales_rots_from_cov(cov: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose Σ = R diag(s^2) R^T -> (scales, quaternion[xyzw])."""
    N = cov.shape[0]
    scales = np.zeros((N, 3), dtype=np.float32)
    rots   = np.zeros((N, 4), dtype=np.float32)  # xyzw
    for i in range(N):
        C = 0.5 * (cov[i] + cov[i].T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        s = np.sqrt(w)
        idx = np.argsort(-s)
        s = s[idx]; R = V[:, idx]
        scales[i] = s.astype(np.float32)
        rots[i] = _quat_from_rotmat(R.astype(np.float32))
    return scales, rots

def _default_colors_opacity(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    colors = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.7  # neutral gray
    opacity = np.ones((xyz.shape[0], 1), dtype=np.float32)
    return colors, opacity

def _project_to_screen(xyz: np.ndarray, proj_T: np.ndarray, W: int, H: int):
    """Project world xyz with full world->clip transform (proj_T.T) to pixel coords.
    Returns (means2D[N,2], valid[N]).
    - proj_T should be (P @ W2C)^T (what we pass to GaussianRasterizer).
    - We implement standard GL-style mapping: ndc in [-1,1]^2 -> pixels.
    """
    full = proj_T.T.astype(np.float32)    # (4,4) world->clip
    xyz1 = np.concatenate([xyz.astype(np.float32), np.ones((len(xyz),1), np.float32)], axis=1)  # (N,4)
    clip = (xyz1 @ full.T)                 # (N,4)
    w = clip[:,3:4]
    # Guard against zero/negatives
    valid = np.isfinite(clip).all(axis=1)
    valid &= (w[:,0] > 0)                  # points in front of camera for our projection
    ndc = np.empty((len(xyz), 2), dtype=np.float32)
    ndc[:] = np.nan
    ndc[valid, 0] = clip[valid,0] / w[valid,0]
    ndc[valid, 1] = clip[valid,1] / w[valid,0]
    # Map to pixels
    u = (ndc[:,0] * 0.5 + 0.5) * float(W)
    v = (-ndc[:,1] * 0.5 + 0.5) * float(H)
    means2D = np.stack([u, v], axis=1).astype(np.float32)
    # Replace NaNs with sentinel far-offscreen to avoid kernel issues
    bad = ~np.isfinite(means2D).all(axis=1)
    means2D[bad] = np.array([-1e6, -1e6], dtype=np.float32)
    return means2D, valid

def _to_np_image(color_t: "torch.Tensor") -> np.ndarray:
    """(C,H,W) or (H,W,C) torch tensor -> (H,W,3) NumPy in [0,1]."""
    import torch
    c = color_t
    if c.ndim == 3 and c.shape[0] in (3,4):   # CHW
        c = c.clamp(0,1).permute(1,2,0).contiguous()
    elif c.ndim == 3 and c.shape[-1] in (3,4): # HWC
        c = c.clamp(0,1).contiguous()
    else:
        raise ValueError("Unexpected color tensor shape: %r" % (tuple(c.shape),))
    if c.shape[-1] == 4:
        c = c[...,:3]
    return c.detach().cpu().numpy()

def _to_np_2d(t: "torch.Tensor") -> np.ndarray:
    """(H,W) or (1,H,W) torch tensor -> (H,W) NumPy float32."""
    import torch
    if t.ndim == 3 and t.shape[0] == 1:  # (1,H,W) -> (H,W)
        t = t[0]
    elif t.ndim == 3 and t.shape[-1] == 1:  # (H,W,1) -> (H,W)
        t = t[...,0]
    assert t.ndim == 2, f"Expected 2D map, got {tuple(t.shape)}"
    return t.detach().cpu().float().numpy()

def _stats_of_tensor(t: "torch.Tensor") -> Tuple[float, float, float]:
    """Return (min,max,mean) in float for debugging."""
    import torch
    mn = float(t.min().detach().cpu().item())
    mx = float(t.max().detach().cpu().item())
    mean = float(t.mean().detach().cpu().item())
    return mn, mx, mean

def _env_debug() -> bool:
    return os.environ.get("GS_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")

def _debug_print(*a, **k):
    if _env_debug():
        print(*a, **k)

# ---- Output parsing ----------------------------------------------------------
def _parse_rasterizer_outputs(out: Any, H: int, W: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Accepts whatever the rasterizer returned (tensor or tuple/list) and
    returns (rgb[H,W,3], depth[H,W] or None, alpha[H,W] or None).
    Heuristics:
      - color: first tensor with 3 or 4 channels (CHW or HWC).
      - depth: 2D map where max>1.05 OR min<0  (meters or non-normalized). If none, the
               2D map with the **largest dynamic range** is used as normalized depth.
      - alpha: 2D map fully in [0,1] with non-trivial range.
      - If color has 4 channels, alpha is the 4th channel.
    """
    import torch
    # Normalize 'vals' to a list of torch Tensors or Nones
    if isinstance(out, (list, tuple)):
        vals: List[Any] = list(out)
    else:
        vals = [out]

    # Debug dump
    _debug_print(f"[3DGS] Output tuple length: {len(vals)}")
    for i,v in enumerate(vals):
        if torch.is_tensor(v):
            try:
                mn,mx,me = _stats_of_tensor(v)
                _debug_print(f"   o{i}: shape={tuple(v.shape)} dtype={v.dtype} min={mn} max={mx} mean={me}")
            except Exception:
                _debug_print(f"   o{i}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            _debug_print(f"   o{i}: <not tensor>")

    # 1) Pick color
    color_t = None
    for v in vals:
        if isinstance(v, torch.Tensor) and v.ndim == 3 and (v.shape[0] in (3,4) or v.shape[-1] in (3,4)):
            color_t = v
            break
    if color_t is None:
        raise RuntimeError("Rasterizer did not return a color tensor.")
    rgb = _to_np_image(color_t)

    # If RGBA, extract alpha from the 4th channel
    alpha: Optional[np.ndarray] = None
    if color_t.ndim == 3 and (color_t.shape[0] == 4 or color_t.shape[-1] == 4):
        if color_t.shape[0] == 4:  # CHW
            alpha = _to_np_2d(color_t[3:4, ...])
        else:  # HWC
            alpha = _to_np_2d(color_t[..., 3:4])
        _debug_print("[3DGS] Alpha extracted from RGBA color channel.")

    # 2) Scan 2D maps to find depth/alpha candidates
    twoD: List[Tuple[int, "torch.Tensor"]] = []
    for i,v in enumerate(vals):
        if isinstance(v, torch.Tensor):
            if (v.ndim == 2) or (v.ndim == 3 and (v.shape[0] == 1 or v.shape[-1] == 1)):
                twoD.append((i,v))

    depth: Optional[np.ndarray] = None
    alpha_from_map: Optional[np.ndarray] = None
    depth_scores: List[Tuple[float,int]] = []  # (score, idx) negative is better

    for i,v in twoD:
        mn, mx, me = _stats_of_tensor(v)
        # Candidate alpha: bounded to [0,1] and non-trivial
        if alpha is None and 0.0 <= mn and mx <= 1.0 and (mx - mn) > 1e-3:
            alpha_from_map = _to_np_2d(v)
            _debug_print(f"[3DGS] Alpha candidate at index {i} (0..1).")
        # Candidate depth: outside [0,1] OR large dynamic range
        score = 0.0
        if (mx > 1.05) or (mn < 0.0):
            score = -(mx - mn)  # prefer larger spread
            depth_scores.append((score, i))
        else:
            # keep as weak candidate based on dynamic range
            score = -(mx - mn)
            depth_scores.append((score, i))

    if depth_scores:
        # best candidate (most negative score = largest spread)
        depth_idx = sorted(depth_scores)[0][1]
        v = [v for (i,v) in twoD if i == depth_idx][0]
        depth = _to_np_2d(v)
        mn, mx, _ = _stats_of_tensor(v)
        if mx > 1.05 or mn < 0.0:
            _debug_print(f"[3DGS] Depth (likely meters) detected at index {depth_idx}.")
        else:
            _debug_print(f"[3DGS] Depth (normalized 0..1) selected at index {depth_idx}.")

    # Finalize alpha
    if alpha is None:
        alpha = alpha_from_map
    if alpha is None:
        # Safe fallback: luminance of color
        lum = 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]
        alpha = np.clip(lum.astype(np.float32), 0.0, 1.0)
        _debug_print("[3DGS] Alpha synthesized from color luminance (fallback).")

    return rgb.astype(np.float32), depth.astype(np.float32) if depth is not None else None, alpha.astype(np.float32) if alpha is not None else None

# ---- Public renderer class ---------------------------------------------------
class GSRenderer3DGS:
    """Thin wrapper that stays persistent across frames.
    Pass matrices built from your camera config (see camera_utils.py).
    """
    def __init__(self,
                 width: int, height: int,
                 tanfovx: float, tanfovy: float,
                 viewmatrix: np.ndarray, projmatrix: np.ndarray,
                 campos: np.ndarray,
                 bg=(1.0, 1.0, 1.0),
                 sh_degree: int = 0,
                 scale_modifier: float = 1.0,
                 prefiltered: bool = False,
                 debug: bool = False,
                 device: str = "cuda"):
        self.device = device
        self.width = int(width); self.height = int(height)
        self._proj_T_np = projmatrix.astype(np.float32).copy()  # numpy copy for CPU projection

        self.settings = GaussianRasterizationSettings(
            image_height=self.height, image_width=self.width,
            tanfovx=float(tanfovx), tanfovy=float(tanfovy),
            bg=_to_torch(np.array(bg, dtype=np.float32), device=self.device),
            scale_modifier=float(scale_modifier),
            viewmatrix=_to_torch(viewmatrix.astype(np.float32), device=self.device),
            projmatrix=_to_torch(projmatrix.astype(np.float32), device=self.device),
            sh_degree=int(sh_degree),
            campos=_to_torch(campos.astype(np.float32), device=self.device),
            prefiltered=bool(prefiltered),
            debug=bool(debug),
        )
        self.rasterizer = GaussianRasterizer(self.settings)

    def render(self,
               xyz: np.ndarray,
               cov: np.ndarray,
               rgb: Optional[np.ndarray] = None,
               opacity: Optional[np.ndarray] = None,
               normals: Optional[np.ndarray] = None,
               prefer_cov_precomp: bool = True,
               return_torch: bool = False,
               render_normal_map: bool = False) -> Dict[str, np.ndarray]:
        """Render one frame given means + 3D covariances.
        Args:
            xyz     : (N,3) world-space means (NumPy or Torch)
            cov     : (N,3,3) or (N,6) world-space covariances (NumPy or Torch)
            rgb     : (N,3) in [0,1], default neutral gray
            opacity : (N,1), default ones
            normals : (N,3) surface normals (optional, for normal map rendering)
            return_torch : if True, returns Torch tensors (keeps gradient)
            render_normal_map : if True, also render a normal map pass
        Returns:
            dict with keys:
              - 'image' : (H,W,3) float32 in [0,1]
              - 'depth' : (H,W) float32 or None
              - 'alpha' : (H,W) float32 in [0,1] or None
              - 'normal_map' : (H,W,3) float32 or None (if render_normal_map=True)
        
        NOTE: For differentiable rendering, pass return_torch=True and
              ensure xyz/cov are Torch tensors with requires_grad=True.
        """
        import torch
        device = self.device
        
        # Check if inputs are already torch tensors
        is_torch_input = torch.is_tensor(xyz)

        # 1) Handle means3D
        if is_torch_input:
            means3D = xyz.to(device) if xyz.device != torch.device(device) else xyz
            # Project to screen (keep on same device if possible, fallback to CPU)
            xyz_np = xyz.detach().cpu().numpy() if not return_torch else xyz.detach().cpu().numpy()
        else:
            xyz_np = np.asarray(xyz)
            means3D = _to_torch(xyz_np.astype(np.float32), device=device)
        
        means2D_np, valid = _project_to_screen(xyz_np, self._proj_T_np, self.width, self.height)
        means2D = _to_torch(means2D_np, device=device)

        # 2) Colors/opacity
        if rgb is None or opacity is None:
            colors, opac = _default_colors_opacity(xyz_np if not is_torch_input else xyz.detach().cpu().numpy())
            if rgb is None: rgb = colors
            if opacity is None: opacity = opac
        
        if torch.is_tensor(rgb):
            colors_t = rgb.to(device) if rgb.device != torch.device(device) else rgb
        else:
            colors_t = _to_torch(np.asarray(rgb).astype(np.float32), device=device)
        
        if torch.is_tensor(opacity):
            opac_t = opacity.to(device) if opacity.device != torch.device(device) else opacity
        else:
            opac_t = _to_torch(np.asarray(opacity).astype(np.float32), device=device)

        # 3) Handle covariance
        if torch.is_tensor(cov):
            cov_np = cov.detach().cpu().numpy()
            cov_is_torch = True
        else:
            cov_np = np.asarray(cov)
            cov_is_torch = False
        
        # 4) Try cov3D_precomp fast-path (if the rasterizer supports it)
        out = None
        if prefer_cov_precomp:
            try:
                cov6 = _as_cov6(cov_np)
                if cov_is_torch and return_torch:
                    # Keep gradient flow
                    cov_t = cov if cov.shape[-1] == 6 else torch.from_numpy(cov6).to(device)
                else:
                    cov_t = _to_torch(cov6, device=device)
                out = self.rasterizer(
                    means3D=means3D, means2D=means2D,
                    opacities=opac_t, colors_precomp=colors_t,
                    cov3D_precomp=cov_t
                )
            except TypeError as e:
                _debug_print(f"[3DGS] cov3D_precomp path failed: {e}")
                out = None
            except Exception as e:
                _debug_print(f"[3DGS] cov3D_precomp path error: {e}")
                out = None

        # 5) Fallback: decompose Σ -> scales, rotations (quaternion xyzw)
        scales_t = None
        rots_t = None
        if out is None:
            if cov_np.ndim == 2 and cov_np.shape[1] == 6:
                # Reconstruct 3x3 for decomposition
                xx,xy,xz,yy,yz,zz = [cov_np[:,i] for i in range(6)]
                C = np.zeros((len(cov_np),3,3), dtype=np.float32)
                C[:,0,0]=xx; C[:,0,1]=xy; C[:,0,2]=xz
                C[:,1,0]=xy; C[:,1,1]=yy; C[:,1,2]=yz
                C[:,2,0]=xz; C[:,2,1]=yz; C[:,2,2]=zz
                cov3 = C
            else:
                cov3 = cov_np.astype(np.float32)
            scales, rots_xyzw = _scales_rots_from_cov(cov3.astype(np.float32))
            scales_t = _to_torch(scales, device=device)
            rots_t   = _to_torch(rots_xyzw, device=device)  # xyzw convention
            out = self.rasterizer(
                means3D=means3D, means2D=means2D,
                opacities=opac_t, colors_precomp=colors_t,
                scales=scales_t, rotations=rots_t
            )

        # 6) Render normal map if requested
        normal_map_out = None
        if render_normal_map and normals is not None:
            # Convert normals from [-1,1] to [0,1] for RGB rendering
            if torch.is_tensor(normals):
                normals_rgb_t = (normals + 1.0) * 0.5
                normals_rgb_t = torch.clamp(normals_rgb_t, 0.0, 1.0).to(device)
            else:
                normals_np = np.asarray(normals).astype(np.float32)
                normals_rgb_np = np.clip((normals_np + 1.0) * 0.5, 0.0, 1.0)
                normals_rgb_t = _to_torch(normals_rgb_np, device=device)
            
            # Always use scales/rotations for normal map rendering (more compatible)
            try:
                # Need to prepare scales/rotations if not already done
                if scales_t is None or rots_t is None:
                    if cov_np.ndim == 2 and cov_np.shape[1] == 6:
                        xx,xy,xz,yy,yz,zz = [cov_np[:,i] for i in range(6)]
                        C = np.zeros((len(cov_np),3,3), dtype=np.float32)
                        C[:,0,0]=xx; C[:,0,1]=xy; C[:,0,2]=xz
                        C[:,1,0]=xy; C[:,1,1]=yy; C[:,1,2]=yz
                        C[:,2,0]=xz; C[:,2,1]=yz; C[:,2,2]=zz
                        cov3 = C
                    else:
                        cov3 = cov_np.astype(np.float32)
                    scales, rots_xyzw = _scales_rots_from_cov(cov3.astype(np.float32))
                    scales_t = _to_torch(scales, device=device)
                    rots_t = _to_torch(rots_xyzw, device=device)
                
                # Render with scales/rotations (more compatible than cov3D_precomp)
                normal_map_out = self.rasterizer(
                    means3D=means3D, means2D=means2D,
                    opacities=opac_t, colors_precomp=normals_rgb_t,
                    scales=scales_t, rotations=rots_t
                )
            except Exception as e:
                _debug_print(f"[3DGS] Normal map rendering failed: {e}")
                import traceback
                traceback.print_exc()
                normal_map_out = None

        # 7) Return torch tensors or numpy arrays
        if return_torch:
            # Keep as torch tensors for gradient flow
            if isinstance(out, (list, tuple)):
                color_t = out[0] if len(out) > 0 else None
                depth_t = out[1] if len(out) > 1 else None
                alpha_t = out[2] if len(out) > 2 else None
            else:
                color_t = out
                depth_t = None
                alpha_t = None
            
            # Basic shape check
            if color_t is not None and color_t.ndim == 3:
                if color_t.shape[0] in (3, 4):  # CHW
                    image_t = color_t.permute(1, 2, 0)[:, :, :3]
                else:  # HWC
                    image_t = color_t[:, :, :3]
            else:
                image_t = color_t
            
            # Parse normal map
            normal_map_t = None
            if normal_map_out is not None:
                if isinstance(normal_map_out, (list, tuple)):
                    nrm_t = normal_map_out[0] if len(normal_map_out) > 0 else None
                else:
                    nrm_t = normal_map_out
                if nrm_t is not None and nrm_t.ndim == 3:
                    if nrm_t.shape[0] in (3, 4):  # CHW
                        normal_map_t = nrm_t.permute(1, 2, 0)[:, :, :3]
                    else:  # HWC
                        normal_map_t = nrm_t[:, :, :3]
            
            return {
                "image": image_t,  # Torch tensor
                "depth": depth_t,  # Torch tensor or None
                "alpha": alpha_t,  # Torch tensor or None
                "normal_map": normal_map_t,  # Torch tensor or None
            }
        else:
            # Parse outputs into numpy (backward compatible)
            rgb_np, depth_np, alpha_np = _parse_rasterizer_outputs(out, self.height, self.width)
            
            # Parse normal map
            normal_map_np = None
            if normal_map_out is not None:
                normal_map_np, _, _ = _parse_rasterizer_outputs(normal_map_out, self.height, self.width)
            
            return {
                "image": rgb_np,
                "depth": depth_np,
                "alpha": alpha_np,
                "normal_map": normal_map_np,
            }