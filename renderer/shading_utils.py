# shading_utils.py
# -----------------------------------------------------------------------------
# Per-splat lighting utilities for Gaussian Splatting.
# Supports Lambertian and Blinn-Phong shading with directional or point lights.
# NEW: robust normal orientation ("view" / "light") and two-sided shading mode
# to eliminate noisy black blotches when PCA normals flip across the surface.
# All functions are NumPy-only to keep them lightweight and portable.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

def _to_np(x):
    return np.asarray(x, dtype=np.float32)

def _get_light_vectors(xyz: np.ndarray, camera_pos: np.ndarray, light_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (L, V, att) where:
      - L is the incident light direction at each point (unit, from point toward light).
      - V is the view direction (unit, from point toward camera).
      - att is a scalar attenuation factor per point (shape: [N,1]).
    Conventions:
      - Directional light: 'direction' is a vector pointing from the light toward the scene.
                           We use L = -normalize(direction) for shading.
      - Point light     : 'position' is the 3D position of the light in world coordinates.
                           L = normalize(light_pos - point).
    """
    xyz = _to_np(xyz)
    camera_pos = _to_np(camera_pos).reshape(1,3)

    typ = (light_cfg.get('type', 'directional') or 'directional').lower().strip()
    if typ == 'directional':
        direction = _to_np(light_cfg.get('direction', [0.3, -0.5, 0.8])).reshape(1,3)
        L = _normalize(-direction).repeat(xyz.shape[0], axis=0)  # incident direction
        att = np.ones((xyz.shape[0], 1), dtype=np.float32)
    else:  # point
        lp = _to_np(light_cfg.get('position', [2.5, 2.0, -2.0])).reshape(1,3)
        vec = lp - xyz
        dist = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
        L = vec / dist
        # Optional attenuation: 1 / (c0 + c1 d + c2 d^2)
        c0, c1, c2 = _to_np(light_cfg.get('attenuation', [1.0, 0.0, 0.0])).reshape(3,)
        att = 1.0 / (c0 + c1*dist + c2*(dist**2))
        att = att.astype(np.float32)

    V = _normalize(camera_pos - xyz)
    return L.astype(np.float32), V.astype(np.float32), att

def _orient_normals(normals: np.ndarray,
                    L: np.ndarray,
                    V: np.ndarray,
                    mode: str = "view",
                    two_sided: bool = True) -> np.ndarray:
    """
    Make PCA normals more stable by flipping them to face either
    the camera ('view') or the light ('light'). Optionally enable
    two-sided shading.
      - normals: (N,3) raw normals (sign arbitrary from local PCA).
      - L      : (N,3) light direction (unit, from point toward light).
      - V      : (N,3) view  direction (unit, from point toward camera).
      - mode   : 'view' or 'light'
      - two_sided: if True, use |N·L| for diffuse (good for very thin shells).
    Returns: oriented normals (N,3).
    """
    N = _normalize(_to_np(normals))
    m = (mode or "view").lower().strip()
    if m == "light":
        s = np.sign((N * L).sum(axis=1, keepdims=True) + 1e-9)
    else:  # "view"
        s = np.sign((N * V).sum(axis=1, keepdims=True) + 1e-9)
    N = N * s  # flip inconsistent normals

    if two_sided:
        # Two-sided: just ensure N faces the dominant half-space used for view,
        # and later take abs() in ndotl to avoid dark patches from flips.
        pass
    return N.astype(np.float32)

def compute_shading(xyz: np.ndarray,
                    normals: np.ndarray,
                    camera_pos: np.ndarray,
                    light_cfg: dict | None = None,
                    albedo_color=(0.7, 0.7, 0.7),
                    model: str = "phong") -> np.ndarray:
    """
    Compute per-splat RGB in [0,1] with simple local shading.
    - xyz        : (N,3) world positions
    - normals    : (N,3) unit normals (will be normalized + oriented defensively)
    - camera_pos : (3,) camera position
    - light_cfg  : dict with keys below (defaults used if missing)
        type: 'directional' | 'point'
        # directional:
        direction: [x,y,z]  (direction from light toward scene)
        # point:
        position: [x,y,z]   (light world position)
        attenuation: [c0, c1, c2]
        # common:
        color: [r,g,b]      (light color)
        intensity: float
        ambient: float
        diffuse: float
        specular: float
        shininess: float
        # NEW:
        orient: 'view' | 'light'  (flip normals to face view or light)  [default: 'view']
        two_sided: bool            (use |N·L| for diffuse)               [default: True]
        min_ambient: float         (floor on ambient to avoid black)     [default: 0.05]
    - albedo_color: (3,) base albedo per splat (can be scalar or (N,3) broadcastable)
    - model: "lambert" or "phong"
    """
    if light_cfg is None:
        light_cfg = {}
    # Defaults
    light_color = _to_np(light_cfg.get('color', [1.0, 1.0, 1.0])).reshape(1,3)
    intensity   = float(light_cfg.get('intensity', 1.0))
    ka = float(light_cfg.get('ambient', 0.10))
    kd = float(light_cfg.get('diffuse', 0.90))
    ks = float(light_cfg.get('specular', 0.10 if model.lower()=='phong' else 0.0))
    shininess = float(light_cfg.get('shininess', 32.0))
    orient    = (light_cfg.get('orient', 'view') or 'view')
    two_sided = bool(light_cfg.get('two_sided', True))
    ka_floor  = float(light_cfg.get('min_ambient', 0.05))

    xyz = _to_np(xyz)
    albedo = _to_np(albedo_color).reshape(1, -1)
    if albedo.shape[1] != 3:
        albedo = np.tile(albedo, (1,3))
    albedo = albedo.astype(np.float32)

    L, V, att = _get_light_vectors(xyz, _to_np(camera_pos), light_cfg)

    # Robust normals: orient + optional two-sided shading
    N = _orient_normals(_to_np(normals), L, V, mode=orient, two_sided=two_sided)

    # Diffuse term
    ndotl_raw = (N * L).sum(axis=1, keepdims=True)
    if two_sided:
        ndotl = np.abs(ndotl_raw)  # handle local flips (thin shells)
    else:
        ndotl = np.clip(ndotl_raw, 0.0, 1.0)
    diffuse = kd * np.clip(ndotl, 0.0, 1.0)

    # Specular (Blinn-Phong)
    if ks > 0.0 and model.lower() == "phong":
        H = _normalize(L + V)
        ndoth = (N * H).sum(axis=1, keepdims=True)
        if two_sided:
            ndoth = np.abs(ndoth)
        else:
            ndoth = np.clip(ndoth, 0.0, 1.0)
        spec = ks * (ndoth ** shininess)
    else:
        spec = 0.0

    # Combine terms
    light_term = (ka + att * intensity * (diffuse + spec)).astype(np.float32)
    # Apply a small ambient floor to avoid near-zero artifacts from extremely dark patches
    light_term = np.maximum(light_term, ka_floor).astype(np.float32)

    rgb = albedo.reshape(1,3) * light_color * light_term
    rgb = np.clip(rgb, 0.0, 1.0)
    # Broadcast albedo per-splat if needed
    if rgb.shape[0] == 1 and xyz.shape[0] > 1:
        rgb = np.repeat(rgb, xyz.shape[0], axis=0)
    return rgb.astype(np.float32)