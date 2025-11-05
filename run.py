"""
run.py - [CVPR 2026] PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

Main entry point for shape morphing pipeline.

Usage:
    python run.py -c configs/sphere_to_spot.yaml --e2e
    python run.py -c configs/sphere_to_spot.yaml --e2e --png

Output Structure:
output/
    ├── target/          # Target renderings
    └── ep{:03d}/        # Episode results

Author: CHAYO
Version: 3.0 (Modularized - 2361 lines → 250 lines!)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
from pathlib import Path
from typing import Dict

# Core modules
from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from loss import E2ELossManager

# Modularized utilities
from utils.physics_utils import (
    build_opt_input,
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
    extract_target_point_cloud,
)
from utils.rendering_utils import (
    setup_renderer,
    create_target_render,
)
from utils.training_loop import run_e2e_episode


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> tuple:
    """Load YAML configuration."""
    import yaml
    
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    return cfg, config_path.parent


def _safe_deepcopy(obj):
    """
    Deep copy that safely handles torch tensors by skipping them.
    """
    import copy
    import torch
    
    if isinstance(obj, torch.Tensor):
        # Skip tensors (they shouldn't be in config anyway)
        return obj
    elif isinstance(obj, dict):
        return {k: _safe_deepcopy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_deepcopy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_safe_deepcopy(item) for item in obj)
    else:
        try:
            return copy.deepcopy(obj)
        except:
            # If deepcopy fails, return as-is
            return obj


def apply_episode_schedule(cfg: Dict, episode: int) -> Dict:
    """
    Apply episode-specific overrides from episode_schedule section.
    
    Args:
        cfg: Base configuration
        episode: Episode number (0-indexed, will be converted to 1-indexed for matching)
    
    Returns:
        Modified configuration with episode overrides
    """
    schedule = cfg.get('episode_schedule', {})
    if not schedule:
        return cfg
    
    # Find matching schedule entry (0-indexed)
    matched_key = None
    for key in schedule.keys():
        key_str = str(key)
        
        if key_str.isdigit() and int(key_str) == episode:
            matched_key = key
            break
        elif '-' in key_str:
            try:
                start, end = key_str.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start <= episode <= end:
                    matched_key = key
                    break
            except ValueError:
                continue
        elif '+' in key_str:
            try:
                start = int(key_str.replace('+', '').strip())
                if episode >= start:
                    matched_key = key
            except ValueError:
                continue
    
    if matched_key is not None:
        cfg_out = _safe_deepcopy(cfg)
        overrides = schedule[matched_key]
        print(f"\n[Episode Schedule] Applying overrides for ep{episode:03d} (key: {matched_key})")
        _deep_update(cfg_out, overrides)
        return cfg_out
    
    return cfg


def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def validate_config(cfg: Dict) -> None:
    """Basic config validation."""
    required = ['input_mesh_path', 'target_mesh_path']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point."""
    # ════════════════════════════════════════════════════════════════════════════
    # Parse Arguments
    # ════════════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(description='PhysMorph-GS Shape Morphing')
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML')
    parser.add_argument('--png', action='store_true', help='Export PNG images')
    parser.add_argument('--png_dpi', type=int, default=160, help='PNG DPI')
    parser.add_argument('--png_ptsize', type=float, default=0.5, help='Point size')
    
    args = parser.parse_args()
    
    # ════════════════════════════════════════════════════════════════════════════
    # Load Configuration
    # ════════════════════════════════════════════════════════════════════════════
    print("="*80)
    print("PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing")
    print("="*80)
    
    cfg, cfg_dir = load_config(args.config)
    validate_config(cfg)
    
    out_dir = Path(cfg.get("output_dir", "output/"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 🔥 CRITICAL: Pass output_dir to upsampling config for stage exports
    if "upsample" not in cfg:
        cfg["upsample"] = {}
    if "debug" not in cfg["upsample"]:
        cfg["upsample"]["debug"] = {}
    cfg["upsample"]["debug"]["output_dir"] = str(out_dir)
    cfg["upsample"]["output_dir"] = str(out_dir)  # Top-level도 설정
    
    # 🔥 NEW: Pass physics grid parameters to upsampling (SDF bbox 일치)
    sim_cfg = cfg.get("simulation", {})
    cfg["upsample"]["physics_grid"] = {
        "grid_min": sim_cfg.get("grid_min_point", [-16.0, -16.0, -16.0]),
        "grid_max": sim_cfg.get("grid_max_point", [16.0, 16.0, 16.0]),
        "grid_dx": sim_cfg.get("grid_dx", 1.0)
    }
    
    # ════════════════════════════════════════════════════════════════════════════
    # Initialize Physics
    # ════════════════════════════════════════════════════════════════════════════
    print("\n[Init] Building optimization input...")
    opt = build_opt_input(cfg)
    
    print(f"[Init] Loading meshes...")
    print(f"  Input:  {cfg['input_mesh_path']}")
    print(f"  Target: {cfg['target_mesh_path']}")
    
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    
    print("[Init] Calculating point cloud volumes...")
    import diffmpm_bindings
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)
    
    print("[Init] Creating computation graph...")
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)
    tgt = extract_target_point_cloud(target_pc)[0]
    
    # ════════════════════════════════════════════════════════════════════════════
    # Setup Upsampling
    # ════════════════════════════════════════════════════════════════════════════
    print("[Config] Setting up upsampling pipeline...")
    
    rs = default_cfg()
    rs_user = cfg.get("upsample", {}) or {}
    
    # Adapt new config structure to internal format
    adapted_cfg = adapt_config({'upsample': rs_user})
    rs.update(adapted_cfg)
    
    # Show curvature_sigma config if present
    if 'covariance' in rs and 'curvature_sigma' in rs['covariance']:
        cs = rs['covariance']['curvature_sigma']
        print(f"[Config] ✓ Target covariance (curvature-based):")
        print(f"          σ_n0={cs.get('sigma_n0', 0.020):.4f}, σ_t0={cs.get('sigma_t0', 0.030):.4f}")
        print(f"          a={cs.get('a', 3.0):.2f}, b={cs.get('b', 0.5):.2f}, u={cs.get('u', 0.4):.2f}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Setup Rendering
    # ════════════════════════════════════════════════════════════════════════════
    print("[Init] Setting up renderer...")
    cam_cfg = cfg.get("camera", {}) or {}
    render_cfg = cfg.get("render", {}) or {}
    particle_color = render_cfg.get("particle_color", [0.27, 0.51, 0.71])
    
    renderer, view_params = setup_renderer(cam_cfg, render_cfg)
    HAVE_3DGS = renderer is not None
        
    # ════════════════════════════════════════════════════════════════════════════
    # Setup E2E Training (controlled by config only)
    # ════════════════════════════════════════════════════════════════════════════
    enable_e2e = cfg.get("optimization", {}).get("loss", {}).get("enabled", False)
    
    loss_manager = None
    target_render = None
    
    if enable_e2e and HAVE_3DGS:
        print("[E2E] Setting up rendering loss...")
        
        loss_cfg = cfg.get("optimization", {}).get("loss", {})
        loss_manager = E2ELossManager(loss_cfg)
        
        print("[E2E] Creating target rendering...")
        campos = view_params.get('campos')
        target_render = create_target_render(
            target_pc, renderer, rs, campos,
            render_cfg, particle_color, out_dir
        )
        
        if target_render is not None:
            print("✅ [E2E] E2E mode ENABLED")
        else:
            print("⚠️ [E2E] Target rendering failed, E2E disabled")
            enable_e2e = False
    else:
        if not HAVE_3DGS:
            print("[WARN] E2E mode requested but renderer not available")
        print("\n[Mode] Running in PHYSICS-ONLY mode")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Training Loop
    # ════════════════════════════════════════════════════════════════════════════
    ema_state = {}
    
    print(f"\n{'='*80}")
    print(f"Starting training: {opt.num_animations} episodes")
    print(f"{'='*80}\n")
    
    for ep in range(int(opt.num_animations)):
        # Apply episode-specific config overrides
        cfg_ep = apply_episode_schedule(cfg, ep)
        
        # Build episode-specific upsampling config
        rs_ep = default_cfg()
        rs_ep_user = cfg_ep.get("upsample", {}) or {}
        
        adapted_cfg_ep = adapt_config({'upsample': rs_ep_user})
        rs_ep.update(adapted_cfg_ep)
        
        # 🔥 Pass episode number to covariance construction
        if "covariance" not in rs_ep:
            rs_ep["covariance"] = {}
        rs_ep["covariance"]["episode"] = ep
        
        # Debug: Show covariance config for this episode
        if "covariance" in rs_ep:
            cov_ep = rs_ep["covariance"]
            print(f"\n[Episode {ep}] Covariance config:")
            print(f"  sigma0: {cov_ep.get('sigma0', 'NOT SET')}")
            if "curvature_sigma" in cov_ep:
                cs_ep = cov_ep["curvature_sigma"]
                print(f"  curvature_sigma (target): σ_n0={cs_ep.get('sigma_n0'):.3f}, σ_t0={cs_ep.get('sigma_t0'):.3f}")
            else:
                print(f"  curvature_sigma: NOT SET (will use default)")
        
        # Fresh level set for each episode (no advection)
        external_levelset = None
        
        # Run episode
        if enable_e2e and loss_manager is not None and target_render is not None:
            # E2E mode with rendering loss
            num_passes = 3
            num_timesteps = int(opt.num_timesteps)       
            control_stride = int(opt.control_stride) 
            campos = view_params.get('campos')
            
            # Create episode-specific output directory
            ep_dir = out_dir / f"ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[DEBUG] Starting E2E episode {ep} with {num_passes} passes")
            ema_state, episode_losses = run_e2e_episode(
                ep, cg, opt, num_timesteps, control_stride, num_passes,
                rs_ep, ema_state, renderer, loss_manager, target_render,
                view_params, campos, render_cfg, particle_color,
                ep_dir, args.png, tgt,
                cov_module=None,
                cov_optimizer=None,
                external_levelset=external_levelset  # 🔥 Pass level set
            )
            print(f"[DEBUG] E2E episode {ep} completed")
            
            # Print episode summary
            print(f"\n[Summary] Episode {ep} losses:")
            for key, val in episode_losses.items():
                if isinstance(val, (int, float)):
                    print(f"  {key}: {val:.6f}")
        else:
            # Physics-only mode
            print(f"\n[Episode {ep}] Running standard physics optimization...")
            print(f"[DEBUG] Physics-only mode (enable_e2e={enable_e2e}, loss_manager={loss_manager is not None}, target_render={target_render is not None})")
            opt.current_episodes = ep
            cg.run_optimization(opt)
    
    # ════════════════════════════════════════════════════════════════════════════
    # Finalization
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print(f"   Output directory: {out_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
