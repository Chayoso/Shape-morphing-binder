# run.py
# Unified runner that integrates DiffMPM + runtime surface + 3DGS renderer.
# - Safe image writer (imageio -> PIL fallback)
# - cov3D_precomp fast path + fallback
# - Proper means2D computation inside renderer to avoid "white screen"
# - YAML render.num_frames controls how many timesteps are rendered.
import argparse, json
from pathlib import Path
import numpy as np

# Safe image writer: imageio.v2 -> PIL fallback
try:
    import imageio.v2 as iio
    def _save_png(path, img):
        iio.imwrite(str(path), img)
except Exception:
    from PIL import Image
    def _save_png(path, img):
        Image.fromarray(img).save(str(path))

def _save_depth16(path, depth_meters):
    import numpy as _np
    # meters -> millimeters in uint16 PNG (0..65535)
    dmm = _np.clip(_np.nan_to_num(depth_meters, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535)
    dmm = dmm.astype(_np.uint16)
    try:
        iio.imwrite(str(path), dmm)
    except Exception:
        from PIL import Image as _Image
        _Image.fromarray(dmm).save(str(path))

# Add this next to _save_depth16 in run.py
def _save_depth_preview8(path, depth_meters, mode="percentile", p=(1.0, 99.0), near=None, far=None):
    """
    Save a view-friendly 8-bit depth preview by normalizing the float depth map.
    Modes:
      - "percentile": map [p_lo, p_hi] percentiles to [0,255]
      - "linear"    : map [near, far] (meters) to [0,255]
      - "inverse"   : map 1/z into [0,255] using [near, far]
    """
    import numpy as _np
    from imageio.v2 import imwrite as _iow
    d = _np.nan_to_num(_np.asarray(depth_meters, dtype=_np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = d > 0

    if mode == "percentile":
        if _np.any(mask):
            lo, hi = _np.percentile(d[mask], p[0]), _np.percentile(d[mask], p[1])
        else:
            lo, hi = 0.0, 1.0
    elif mode == "inverse":
        if near is None: near = max(1e-3, d[mask].min() if _np.any(mask) else 0.01)
        if far  is None: far  = (d[mask].max() if _np.any(mask) else 10.0)
        z = d.copy(); z[~mask] = 0.0
        inv = _np.where(mask, 1.0/_np.clip(z, near, far), 0.0)
        lo, hi = inv[mask].min(), inv[mask].max()
        d = inv
    else:  # "linear"
        if near is None: near = (d[mask].min() if _np.any(mask) else 0.0)
        if far  is None:  far = (d[mask].max() if _np.any(mask) else 1.0)
        lo, hi = near, far

    s = ( (d - lo) / (max(hi - lo, 1e-6)) )
    s = _np.clip(s, 0.0, 1.0)
    _iow(str(path), (s * 255.0).astype(_np.uint8))


# --- DiffMPM bindings --------------------------------------------------------
try:
    import diffmpm_bindings
    BINDINGS_AVAILABLE = True
except Exception:
    diffmpm_bindings = None
    BINDINGS_AVAILABLE = False

# --- Runtime surface (upsampler) --------------------------------------------
from sampling.runtime_surface import (
    default_cfg,
    synthesize_runtime_surface,
    save_ply_xyz,
    save_gaussians_npz,
    save_comparison_png,
    save_axis_hist_png,
)

# --- 3DGS integration --------------------------------------------------------
try:
    from renderer.camera_utils import make_matrices_from_yaml
except Exception:
    from camera_utils import make_matrices_from_yaml  # type: ignore

try:
    from renderer.renderer import GSRenderer3DGS
except Exception:
    from renderer import GSRenderer3DGS  # type: ignore
    
# --- Lighting & Compositing --------------------------------------------------
try:
    from renderer.shading_utils import compute_shading
except Exception:
    from shading_utils import compute_shading  # type: ignore

try:
    from renderer.composite_utils import composite_with_background
except Exception:
    from composite_utils import composite_with_background  # type: ignore
def _np(x): return np.asarray(x)


def _pick_timesteps(num_layers: int, num_frames: int, schedule: str = "last_n"):
    num_frames = int(max(0, num_frames))
    if num_frames == 0 or num_layers <= 0:
        return []
    num_frames = min(num_frames, num_layers)
    if schedule == "uniform":
        if num_frames == 1:
            return [num_layers - 1]
        xs = np.linspace(0, num_layers - 1, num_frames)
        return sorted(set(int(round(v)) for v in xs))
    start = max(0, num_layers - num_frames)
    return list(range(start, num_layers))


def main():
    import yaml
    ap = argparse.ArgumentParser(description="DiffMPM runtime-surface runner (3DGS integrated)")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--png", action="store_true", help="Export PNGs (comparison, axis hist).")
    ap.add_argument("--png-dpi", type=int, default=160)
    ap.add_argument("--png-ptsize", type=float, default=0.6)
    args = ap.parse_args()

    if not BINDINGS_AVAILABLE:
        raise RuntimeError("C++ diffmpm_bindings are required.")
    
    # -------------------------------------------------------------------------
    # Load YAML config
    # -------------------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_dir = Path(args.config).parent
    out_dir = Path(cfg.get("output_dir", "output/runtime_surface"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build OptInput from YAML
    sim = cfg["simulation"]; run = cfg["optimization"]
    opt = diffmpm_bindings.OptInput()
    opt.mpm_input_mesh_path  = cfg["input_mesh_path"]
    opt.mpm_target_mesh_path = cfg["target_mesh_path"]
    opt.grid_dx = float(sim["grid_dx"])
    opt.grid_min_point = sim["grid_min_point"]; opt.grid_max_point = sim["grid_max_point"]
    opt.points_per_cell_cuberoot = int(sim.get("points_per_cell_cuberoot", 2))
    opt.lam  = float(sim["lam"]); opt.mu = float(sim["mu"]); opt.p_density = float(sim["density"])
    opt.dt   = float(sim["dt"]);  opt.drag = float(sim["drag"]); opt.f_ext = sim["external_force"]
    opt.smoothing_factor = float(sim["smoothing_factor"])
    opt.num_animations = int(run["num_animations"]); opt.num_timesteps = int(run["num_timesteps"])
    opt.control_stride  = int(run["control_stride"])
    opt.max_gd_iters = int(run["max_gd_iters"]); opt.max_ls_iters = int(run["max_ls_iters"])
    opt.initial_alpha = float(run["initial_alpha"]); opt.gd_tol = float(run["gd_tol"])
    opt.current_episodes = 0

    # Create & run comp graph
    # Load point clouds (target once for PNG comparison)
    input_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_input_mesh_path, opt)
    target_pc = diffmpm_bindings.load_point_cloud_from_obj(opt.mpm_target_mesh_path, opt)

    grid_dims = [int((opt.grid_max_point[i] - opt.grid_min_point[i]) / opt.grid_dx) + 1 for i in range(3)]
    input_grid  = diffmpm_bindings.Grid(grid_dims[0], grid_dims[1], grid_dims[2], opt.grid_dx, opt.grid_min_point)
    target_grid = diffmpm_bindings.Grid(grid_dims[0], grid_dims[1], grid_dims[2], opt.grid_dx, opt.grid_min_point)

    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)

    cg = diffmpm_bindings.CompGraph(input_pc, input_grid, target_grid)

    # Extract target positions (PNG compare)
    tgt = _np(diffmpm_bindings.get_positions_from_pc(target_pc))

    # Runtime-surface config
    rs = default_cfg()
    rs_user = cfg.get("sampling", {}).get("runtime_surface", {}) or {}
    rs.update(rs_user)
    rs.setdefault("png", {"enabled": True, "dpi": 160, "ptsize": 0.5})
    if args.png:
        rs["png"]["enabled"] = True
        rs["png"]["dpi"] = args.png_dpi
        rs["png"]["ptsize"] = args.png_ptsize
    
    # Debug: Print sigma0 value
    print(f"[Runtime Surface] sigma0={rs.get('sigma0', 0.02):.4f} M={rs.get('M', 180000)}")

    # 3DGS render config (YAML)
    render_cfg = cfg.get("render", {}) or {}
    num_frames = int(render_cfg.get("num_frames",
                    render_cfg.get("timesteps",
                    render_cfg.get("frames", 0))))
    schedule   = str(render_cfg.get("schedule", "last_n")).lower()
    bg         = render_cfg.get("bg", [1.0, 1.0, 1.0])
    particle_color = render_cfg.get("particle_color", [0.7, 0.7, 0.7])  # default gray
    print(f"[Render] num_frames={num_frames} schedule={schedule} particle_color={particle_color}")

    # Camera from YAML
    cam_cfg = cfg.get("camera", {}) or {}
    W, H, tanfovx, tanfovy, view_T, proj_T, campos = make_matrices_from_yaml(cam_cfg)
    print(f"[3DGS] Camera: W={W} H={H} tanfovx={tanfovx:.6f} tanfovy={tanfovy:.6f} "
          f"campos=({campos[0]:.4f},{campos[1]:.4f},{campos[2]:.4f}) "
          f"znear={cam_cfg.get('znear',0.01)} zfar={cam_cfg.get('zfar',100.0)}")

    # Persistent EMA state for drift suppression across episodes
    ema_state = {}

    # Prepare 3DGS renderer (optional; we skip gracefully on failure)
    try:
        renderer = GSRenderer3DGS(
            W, H, tanfovx, tanfovy, view_T, proj_T, campos,
            bg=tuple(bg), sh_degree=0, scale_modifier=1.0, prefiltered=False, debug=False, device="cuda"
        )
        HAVE_3DGS = True
    except Exception as e:
        print("[WARN] 3DGS renderer failed to initialize. Rendering will be skipped.\n", e)
        HAVE_3DGS = False
        
    # -------------------------------------------------------------------------
    # Main optimization loop
    # -------------------------------------------------------------------------
    png_enabled = rs.get("png", {}).get("enabled", True) or getattr(args, 'png', False)

    for ep in range(int(opt.num_animations)):
        print(f"# ====================== Episode {ep+1}/{opt.num_animations} START ====================== #")
        opt.current_episodes = ep
        cg.run_optimization(opt)
        print("# ====================== OPTIMIZATION ENDED ========================== #")

        # Extract current low-res positions and deformation grads
        last = cg.get_num_layers() - 1
        pc   = cg.get_point_cloud(last)
        x    = _np(pc.get_positions())
        F    = _np(pc.get_def_grads_total())

        # Generate runtime surface
        result = synthesize_runtime_surface(x, F, rs, ema_state=ema_state, seed=1234+ep)
        mu, cov, ema_state = result["points"], result["cov"], result["state"]
        
        # Debug: Check covariance after synthesis
        cov_diag_ep = np.array([cov[i].diagonal() for i in range(len(cov))])
        sigma0_val = float(rs.get('sigma0', 0.02))
        print(f"[Surface] N={len(mu)} cov_diag_mean=[{cov_diag_ep.mean(axis=0)[0]:.6f}, "
              f"{cov_diag_ep.mean(axis=0)[1]:.6f}, {cov_diag_ep.mean(axis=0)[2]:.6f}] "
              f"(sigma0={sigma0_val:.4f}, σ²={sigma0_val**2:.6f})")

        # Save artefacts
        ep_dir = out_dir / f"ep{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # (1) comparison.png & (2) axis_hist.png
        if png_enabled:
            cmp_path = ep_dir / f"ep{ep:03d}_comparison.png"
            png_dpi = rs.get("png", {}).get("dpi", 160)
            png_ptsize = rs.get("png", {}).get("ptsize", 0.5)
            save_comparison_png(cmp_path, current_before=x, current_after=mu, radial_after=mu, target_before=tgt,
                                dpi=png_dpi, ptsize=png_ptsize)
            hist_path = ep_dir / f"ep{ep:03d}_axis_hist.png"
            save_axis_hist_png(hist_path, mu, dpi=png_dpi)

        # (3) summary.json (+ J statistics Report)
        debug = result["debug"]
        J = np.linalg.det(F)  # F is total def-grad from bindings
        Jmin_cfg = float(cfg.get("sampling", {}).get("Jmin_diag", 0.60))
        debug.update({
            "J_min": float(J.min()),
            "J_mean": float(J.mean()),
            "J_frac_lt_Jmin": float((J < Jmin_cfg).mean()),
            "ema_thr_next": ema_state.get("ema_thr"),
            "ema_ratio_next": ema_state.get("ema_ratio"),
        })
        with (ep_dir / f"ep{ep:03d}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)

        # (4) gaussians.npz (with YAML-defined color)
        npz_path = ep_dir / f"ep{ep:03d}_gaussians.npz"
        rgb_mu = np.tile(np.array(particle_color, dtype=np.float32), (len(mu), 1))
        save_gaussians_npz(npz_path, mu, cov, rgb=rgb_mu)

        # (5) surface.ply
        ply_path = ep_dir / f"ep{ep:03d}_surface_{len(mu)}.ply"
        save_ply_xyz(ply_path, mu)

        # (6) 3DGS render per-timestep
        if HAVE_3DGS and num_frames > 0:
            num_layers = cg.get_num_layers()
            indices = _pick_timesteps(num_layers, num_frames, schedule=schedule)
            rdir = ep_dir / "renders"
            rdir.mkdir(parents=True, exist_ok=True)

            for t in indices:
                pc_t = cg.get_point_cloud(t)
                x_t  = _np(pc_t.get_positions())
                F_t  = _np(pc_t.get_def_grads_total())

                result_t = synthesize_runtime_surface(x_t, F_t, rs, ema_state=ema_state, seed=1000*ep + t)
                mu_t, cov_t = result_t["points"], result_t["cov"]
                # Optional normals for shading
                nrm_t = result_t.get("normals", None)

                # --- Shading setup (from YAML 'render.lighting') ---
                light_cfg = render_cfg.get("lighting", {}) if isinstance(render_cfg, dict) else {}
                shading_model = str(light_cfg.get("model", "phong")).lower()
                albedo = np.array(particle_color, dtype=np.float32)

                # Compute shading colors (per-splat)
                try:
                    rgb_t = compute_shading(mu_t, nrm_t if nrm_t is not None else np.zeros_like(mu_t),
                                            camera_pos=campos, light_cfg=light_cfg, albedo_color=albedo,
                                            model=shading_model)
                except Exception as _e:
                    print(f"[WARN] Shading failed at t={t}: {_e}. Falling back to flat color.")
                    rgb_t = np.tile(albedo[None,:], (len(mu_t), 1)).astype(np.float32)

                try:
                    # Colors computed by shading above
                    # Debug: Check covariance matrix statistics
                    cov_diag = np.array([cov_t[i].diagonal() for i in range(len(cov_t))])
                    cov_mean_diag = cov_diag.mean(axis=0)
                    print(f"  [t={t}] N={len(mu_t)} cov_diag_mean=[{cov_mean_diag[0]:.6f}, {cov_mean_diag[1]:.6f}, {cov_mean_diag[2]:.6f}]")
                    
                    out = renderer.render(mu_t, cov_t, rgb=rgb_t, prefer_cov_precomp=True)
                    img = out["image"].astype(np.float32)
                    depth = out.get("depth", None)
                    alpha = out.get("alpha", None)

                    # Background handling (paths are relative to YAML file)
                    bg_cfg = render_cfg.get("background", {}) if isinstance(render_cfg, dict) else {}
                    bg_img = None; bg_depth = None
                    try:
                        import imageio.v2 as _iio
                        from pathlib import Path as _Path
                        if isinstance(bg_cfg, dict) and bg_cfg.get("image", None):
                            _img_path = _Path(bg_cfg["image"])
                            if not _img_path.is_absolute():
                                _img_path = (cfg_dir / _img_path)
                            bg_img = _iio.imread(str(_img_path)).astype("float32") / 255.0
                        if isinstance(bg_cfg, dict) and bg_cfg.get("depth", None):
                            depth_scale = float(bg_cfg.get("depth_scale", 1.0))
                            _depth_path = _Path(bg_cfg["depth"])
                            if not _depth_path.is_absolute():
                                _depth_path = (cfg_dir / _depth_path)
                            if str(_depth_path).lower().endswith(".npy"):
                                bg_depth = np.load(str(_depth_path)).astype(np.float32) * depth_scale
                            else:
                                bgd = _iio.imread(str(_depth_path))
                                if bgd.ndim == 3:
                                    bgd = bgd[...,0]
                                bg_depth = bgd.astype("float32") * depth_scale
                    except Exception as _e:
                        print(f"[WARN] Background load failed: {_e}")

                    # Composite
                    comp = composite_with_background(img, alpha, depth, bg_img, bg_depth)

                    # Save outputs
                    frame_path = rdir / f"frame_{t:04d}.png"
                    _save_png(frame_path, (np.clip(comp,0,1)*255).astype(np.uint8))
                    if depth is not None:
                        _save_depth16(rdir / f"frame_{t:04d}_depth.png", depth.astype(np.float32))  # 16-bit, millimeters
                        # NEW: human-friendly preview (auto normalized 8-bit)
                        _save_depth_preview8(rdir / f"frame_{t:04d}_depth_preview.png",
                                             depth.astype(np.float32), mode=render_cfg.get("depth_preview", {}).get("mode", "percentile"),
                                             p=tuple(render_cfg.get("depth_preview", {}).get("percentiles", [1.0, 99.0])),
                                             near=render_cfg.get("depth_preview", {}).get("near", None),
                                             far=render_cfg.get("depth_preview", {}).get("far", None))
                    if alpha is not None:
                        a8 = (np.clip(alpha,0,1)*255).astype(np.uint8)
                        _save_png(rdir / f"frame_{t:04d}_alpha.png", a8)
                except Exception as e:
                    print(f"[WARN] 3DGS render failed at t={t}: {e}")
                    
        # Promote for next episode
        if ep < int(opt.num_animations) - 1:
            cg.promote_last_as_initial()

    print("All episodes finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())