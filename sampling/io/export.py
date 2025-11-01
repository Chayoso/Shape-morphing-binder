"""File export utilities (PNG, PLY, NPZ)."""

import numpy as np
from pathlib import Path
from ..utils.utils import as_numpy


def setup_matplotlib():
    """Setup matplotlib for non-interactive mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def compute_plot_bounds(points_list):
    """Compute unified plot bounds for multiple point clouds."""
    all_points = np.concatenate([p for p in points_list if p is not None and p.size > 0], axis=0)
    if all_points.size == 0:
        return None
    
    mins, maxs = all_points.min(0), all_points.max(0)
    rng = (maxs - mins).max() * 0.5
    mid = (maxs + mins) * 0.5
    return mid, rng


def set_axis_limits(ax, mid, rng):
    """Set 3D axis limits with uniform scaling."""
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def save_comparison_png(path, current_before=None, current_after=None, radial_after=None, target_before=None, dpi=160, ptsize=0.5):
    """Save 3-panel comparison visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    X = as_numpy(current_before) if current_before is not None else None
    PU = as_numpy(current_after) if current_after is not None else None
    
    plt = setup_matplotlib()
    
    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5, 0.5, "No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    
    if X is not None and X.size > 0:
        ax1.scatter(X[:,0], X[:,1], X[:,2], s=ptsize, alpha=0.6)
    ax1.set_title("Current (before upsampling)")
    
    ax2.scatter(PU[:,0], PU[:,1], PU[:,2], s=ptsize, alpha=0.6)
    ax2.set_title("Current (after upsampling)")
    
    c = PU.mean(0)
    r = np.linalg.norm(PU - c[None,:], axis=1)
    sc = ax3.scatter(PU[:,0], PU[:,1], PU[:,2], c=r, s=ptsize, alpha=0.6, cmap="viridis")
    ax3.set_title("Radial color (after upsampling)")
    fig.colorbar(sc, ax=ax3, shrink=0.6)
    
    points_list = [X, PU]
    bounds = compute_plot_bounds(points_list)
    if bounds is not None:
        mid, rng = bounds
        for ax in (ax1, ax2, ax3):
            set_axis_limits(ax, mid, rng)
    
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_axis_hist_png(path, pts, dpi=160):
    """Save axis histogram visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    PU = as_numpy(pts)
    plt = setup_matplotlib()
    
    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5, 0.5, "No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    titles = ["X-Axis Distribution", "Y-Axis Distribution", "Z-Axis Distribution"]
    labels = ["X", "Y", "Z"]
    
    for j, (ax, title, label) in enumerate(zip(axes, titles, labels)):
        ax.hist(PU[:, j], bins=48, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(f"{label} Coordinate")
        ax.set_ylabel("Count")
    
    fig.suptitle("Runtime Surface (Axis Distribution)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_ply_xyz(path: Path, xyz: np.ndarray):
    """Save point cloud in PLY format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, xyz, fmt="%.6f")


def save_gaussians_npz(path: Path, xyz: np.ndarray, cov: np.ndarray, rgb=None, opacity=None):
    """Save Gaussian splatting data in NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if rgb is None:
        rgb = np.ones_like(xyz, dtype=np.float32)
    if opacity is None:
        opacity = np.ones((len(xyz), 1), dtype=np.float32)
    
    np.savez_compressed(
        path, 
        xyz=xyz.astype(np.float32),
        cov=cov.astype(np.float32),
        rgb=rgb.astype(np.float32),
        opacity=opacity.astype(np.float32)
    )


def save_anchor_visualization(
    path,
    x_low,
    anchors,
    surf_prob=None,
    anchor_density=None,
    dpi=160,
    ptsize=1.0
):
    """
    Anchor visualization: Original points vs sampled anchors
    
    Args:
        path: Save path
        x_low: (N, 3) Original anchor points
        anchors: (M, 3) Sampled anchors
        surf_prob: (N,) Surface probability (optional, for coloring)
        anchor_density: (N,) Anchor density ρ (optional, for coloring)
        dpi: Image resolution
        ptsize: Point size
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    X = as_numpy(x_low)
    A = as_numpy(anchors)
    
    plt = setup_matplotlib()
    
    if X is None or X.size == 0:
        fig = plt.figure(figsize=(3,3))
        fig.text(0.5, 0.5, "No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    
    # 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    
    # 1) Original anchors (colored by surface probability)
    ax1 = fig.add_subplot(221, projection="3d")
    if surf_prob is not None:
        prob = as_numpy(surf_prob)
        sc1 = ax1.scatter(X[:,0], X[:,1], X[:,2], c=prob, s=ptsize*2, 
                         alpha=0.6, cmap="coolwarm", vmin=0, vmax=prob.max())
        fig.colorbar(sc1, ax=ax1, shrink=0.6, label="Surface Probability")
        ax1.set_title(f"Original Anchors (N={len(X):,})\nColored by Surface Prob")
    else:
        ax1.scatter(X[:,0], X[:,1], X[:,2], s=ptsize*2, alpha=0.6, c='blue')
        ax1.set_title(f"Original Anchors (N={len(X):,})")
    
    # 2) Original anchors (colored by anchor density)
    ax2 = fig.add_subplot(222, projection="3d")
    if anchor_density is not None:
        rho = as_numpy(anchor_density)
        rho_min, rho_max = rho.min(), rho.max()
        sc2 = ax2.scatter(X[:,0], X[:,1], X[:,2], c=rho, s=ptsize*2,
                         alpha=0.6, cmap="viridis", vmin=rho_min, vmax=rho_max)
        fig.colorbar(sc2, ax=ax2, shrink=0.6, label="Anchor Density (ρ)")
        ax2.set_title(f"Original Anchors (N={len(X):,})\nColored by Anchor Density")
        
        # Display density statistics
        rho_mean = rho.mean()
        ax2.text2D(0.05, 0.95, f"Density ρ: mean={rho_mean:.3f}, [{rho_min:.3f}, {rho_max:.3f}]",
                  transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.scatter(X[:,0], X[:,1], X[:,2], s=ptsize*2, alpha=0.6, c='blue')
        ax2.set_title(f"Original Anchors (N={len(X):,})")
    
    # 3) Sampled anchors
    ax3 = fig.add_subplot(223, projection="3d")
    if A is not None and A.size > 0:
        ax3.scatter(A[:,0], A[:,1], A[:,2], s=ptsize, alpha=0.5, c='red')
        ax3.set_title(f"Sampled Anchors (M={len(A):,})\nUpsampling Factor: {len(A)/len(X):.1f}×")
    else:
        ax3.text2D(0.5, 0.5, "No sampled anchors", ha="center")
        ax3.set_title("Sampled Anchors")
    
    # 4) Overlay comparison
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.scatter(X[:,0], X[:,1], X[:,2], s=ptsize*2, alpha=0.3, c='blue', label='Original')
    if A is not None and A.size > 0:
        ax4.scatter(A[:,0], A[:,1], A[:,2], s=ptsize*0.5, alpha=0.3, c='red', label='Sampled')
    ax4.set_title("Overlay Comparison")
    ax4.legend(loc='upper right')
    
    # Unified bounds
    points_list = [X]
    if A is not None and A.size > 0:
        points_list.append(A)
    bounds = compute_plot_bounds(points_list)
    if bounds is not None:
        mid, rng = bounds
        for ax in (ax1, ax2, ax3, ax4):
            set_axis_limits(ax, mid, rng)
    
    fig.suptitle("Anchor Sampling Visualization", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Anchor visualization saved: {path}")


def save_stage_progression(
    base_dir,
    episode,
    stage_data,
    dpi=160,
    ptsize=0.5  # Default point size for visualization
):
    """
    Save visualization of upsampling progression across all stages.
    
    Creates a base_dir/episode_XXX/ folder containing:
    - STAGE1.png: Anchor points with surface probability
    - STAGE2.png: Anchor-density map (anchors only, no upsampled points)
    - STAGE3.png: After importance sampling (upsampled)
    - STAGE4.png: After Taubin smoothing
    - STAGE5.png: After normal smoothing
    - STAGE6.png: Final output with covariances
    - progression.png: All stages in one image
    
    Args:
        base_dir: Base output directory (e.g., "debug")
        episode: Episode number (0-indexed)
        stage_data: Dictionary containing stage outputs:
            {
                'stage1': {'points': (N,3), 'surf_prob': (N,), 'normals': (N,3)},
                'stage2': {'points': (N,3), 'rho_anchor': (N,)},
                'stage3': {'points': (M,3), 'anchors': (M,3)},
                'stage4': {'points': (M,3)},
                'stage5': {'points': (M,3), 'normals': (M,3)},
                'stage6': {'points': (M,3), 'cov': (M,3,3)},
            }
        dpi: Image resolution
        ptsize: Point size
    """
    base_dir = Path(base_dir)
    # Use "target" folder for episode -1, otherwise epXXX
    if episode == -1:
        ep_dir = base_dir
    else:
        ep_dir = base_dir / f"ep{episode:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    
    plt = setup_matplotlib()
    
    # Extract stage data
    stages = []
    stage_names = []
    
    if 'stage1' in stage_data:
        stages.append(stage_data['stage1'])
        stage_names.append("STAGE 1: Surface Detection")
    if 'stage2' in stage_data:
        stages.append(stage_data['stage2'])
        stage_names.append("STAGE 2: Anchor-Density Map")
    if 'stage3' in stage_data:
        stages.append(stage_data['stage3'])
        stage_names.append("STAGE 3: Importance Sampling")
    if 'stage4' in stage_data:
        stages.append(stage_data['stage4'])
        stage_names.append("STAGE 4: Taubin Smoothing")
    if 'stage5' in stage_data:
        stages.append(stage_data['stage5'])
        stage_names.append("STAGE 5: Normal Smoothing")
    if 'stage6' in stage_data:
        stages.append(stage_data['stage6'])
        stage_names.append("STAGE 6: Covariance Construction")
    
    n_stages = len(stages)
    if n_stages == 0:
        print("⚠️ No stage data provided")
        return
    
    # Compute unified bounds for all stages
    all_points = []
    for stage in stages:
        pts = as_numpy(stage.get('points'))
        if pts is not None and pts.size > 0:
            all_points.append(pts)
    
    if len(all_points) == 0:
        print("⚠️ No valid points in any stage")
        return
    
    bounds = compute_plot_bounds(all_points)
    if bounds is None:
        print("⚠️ Could not compute plot bounds")
        return
    
    mid, rng = bounds
    
    # 🔥 SPECIAL HANDLING: STAGE1 with volume + surface side-by-side
    stage1_saved = False
    if 'stage1' in stage_data:
        stage1 = stage_data['stage1']
        pts_final = as_numpy(stage1.get('points'))
        planarity = as_numpy(stage1.get('planarity'))
        anisotropy = as_numpy(stage1.get('anisotropy'))
        z_s = as_numpy(stage1.get('z_s'))
        rho_gate = as_numpy(stage1.get('rho_gate'))
        p_surf_raw = as_numpy(stage1.get('p_surf_raw'))
        spacing = as_numpy(stage1.get('spacing'))
        
        # Get STAGE2 data for w_density and π
        stage2 = stage_data.get('stage2', {})
        w_density = as_numpy(stage2.get('w_den'))
        pi = as_numpy(stage2.get('pi'))
        
        if pts_final is not None and pts_final.size > 0:
            import numpy as np
            
            # STAGE1: 2x2 Z-Score → Boost Visualization
            # planarity, anisotropy, z_s, rho_gate(top 10%), p_surf_raw
            fig = plt.figure(figsize=(24, 14))
            
            # Panel 1: Planarity (s)
            ax1 = fig.add_subplot(221, projection="3d")
            if planarity is not None and planarity.size > 0:
                plan_min, plan_max = planarity.min(), planarity.max()
                plan_mean = planarity.mean()
                sc1 = ax1.scatter(pts_final[:,0], pts_final[:,1], pts_final[:,2], c=planarity, s=ptsize*2,
                                 alpha=0.6, cmap="viridis", vmin=plan_min, vmax=plan_max)
                fig.colorbar(sc1, ax=ax1, shrink=0.6, label="s")
                ax1.set_title(f"[1] Planarity s=(λ₁+λ₂)/Σλ\nmean={plan_mean:.3f}, range=[{plan_min:.3f}, {plan_max:.3f}]", fontsize=10)
            set_axis_limits(ax1, mid, rng)
            
            # Panel 2: Anisotropy (ρ)
            ax2 = fig.add_subplot(222, projection="3d")
            if anisotropy is not None and anisotropy.size > 0:
                aniso_min, aniso_max = anisotropy.min(), anisotropy.max()
                aniso_mean = anisotropy.mean()
                aniso_p90 = np.percentile(anisotropy, 90)
                sc2 = ax2.scatter(pts_final[:,0], pts_final[:,1], pts_final[:,2], c=anisotropy, s=ptsize*2,
                                 alpha=0.6, cmap="plasma", vmin=aniso_min, vmax=aniso_max)
                fig.colorbar(sc2, ax=ax2, shrink=0.6, label="ρ")
                ax2.set_title(f"[2] Anisotropy ρ=(λ₂-λ₁)/λ₂\nmean={aniso_mean:.3f}, P90={aniso_p90:.3f}", fontsize=10)
            set_axis_limits(ax2, mid, rng)
            
            # Panel 3: z_s (Z-Score of Planarity)
            ax3 = fig.add_subplot(223, projection="3d")
            if z_s is not None and z_s.size > 0:
                z_min, z_max = z_s.min(), z_s.max()
                z_mean = z_s.mean()
                z_above_tau = (z_s > 0).mean()  # Assuming tau calibrated around 0
                sc3 = ax3.scatter(pts_final[:,0], pts_final[:,1], pts_final[:,2], c=z_s, s=ptsize*2,
                                 alpha=0.6, cmap="RdBu_r", vmin=z_min, vmax=z_max)
                fig.colorbar(sc3, ax=ax3, shrink=0.6, label="z_s")
                ax3.set_title(f"[3] z_s = (s - μ_s)/σ_s\nmean={z_mean:.3f}, >0={z_above_tau:.1%}", fontsize=10)
            set_axis_limits(ax3, mid, rng)
            
            # Panel 4: p_surf_raw (Final Surface Probability)
            ax4 = fig.add_subplot(224, projection="3d")
            if p_surf_raw is not None and p_surf_raw.size > 0:
                psurf_min, psurf_max = p_surf_raw.min(), p_surf_raw.max()
                psurf_mean = p_surf_raw.mean()
                psurf_above_50 = (p_surf_raw > 0.5).mean()
                sc4 = ax4.scatter(pts_final[:,0], pts_final[:,1], pts_final[:,2], c=p_surf_raw, s=ptsize*2,
                                 alpha=0.6, cmap="coolwarm", vmin=psurf_min, vmax=psurf_max)
                fig.colorbar(sc4, ax=ax4, shrink=0.6, label="p_surf_raw")
                ax4.set_title(f"[4] p_surf_raw (Final)\nsigmoid((z_s-τ)/γ) + w_curv·ρ̂\nmean={psurf_mean:.3f}, >0.5={psurf_above_50:.1%}", fontsize=10)
            set_axis_limits(ax4, mid, rng)
            
            # Save
            stage1_path = ep_dir / "STAGE1.png"
            fig.tight_layout()
            fig.savefig(stage1_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Saved STAGE1 (2x2 PCA → Z-Score → p_surf_raw): {stage1_path}")
            if planarity is not None:
                print(f"    [1] Planarity (s):    mean={plan_mean:.4f}, range=[{plan_min:.3f}, {plan_max:.3f}]")
            if anisotropy is not None:
                print(f"    [2] Anisotropy (ρ):   mean={aniso_mean:.4f}, P90={aniso_p90:.3f}")
            if z_s is not None:
                print(f"    [3] z_s:              mean={z_mean:.3f}, >0={z_above_tau:.1%}")
            if p_surf_raw is not None:
                print(f"    [4] p_surf_raw:       mean={psurf_mean:.3f}, >0.5={psurf_above_50:.1%}")
            stage1_saved = True
    
    # Create individual stage visualizations (skip stage1 if already saved)
    for i, (stage, name) in enumerate(zip(stages, stage_names), start=1):
        if i == 1 and stage1_saved:
            continue  # Skip stage1, already saved
            
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        pts = as_numpy(stage.get('points'))
        if pts is None or pts.size == 0:
            ax.text2D(0.5, 0.5, "No points", transform=ax.transAxes, ha="center")
        else:
            # Color by different properties per stage
            if False:  # i == 1 block removed
                pass
            elif i == 2:
                # Stage 2: Multi-Scale Density Weight (w_den)
                w_den = as_numpy(stage.get('w_den'))
                pi_stage2 = as_numpy(stage.get('pi'))
                
                if w_den is not None and w_den.size > 0:
                    wden_min, wden_max = w_den.min(), w_den.max()
                    wden_mean = w_den.mean()
                    print(f"    [STAGE2] w_density range: [{wden_min:.4f}, {wden_max:.4f}], mean: {wden_mean:.4f}")
                    
                    # Create 1x2 visualization: w_density + π
                    plt.close(fig)
                    fig = plt.figure(figsize=(18, 7))
                    
                    # Left: w_density
                    ax_left = fig.add_subplot(121, projection="3d")
                    sc_left = ax_left.scatter(pts[:,0], pts[:,1], pts[:,2], c=w_den, s=ptsize*2,
                                   alpha=0.4, cmap="viridis", vmin=wden_min, vmax=wden_max)
                    fig.colorbar(sc_left, ax=ax_left, shrink=0.6, label="w_density")
                    ax_left.set_title(f"w_density (Multi-Scale)\nharmonic_mean(ρ₈, ρ₃₂) → sigmoid\nmean={wden_mean:.3f}, range=[{wden_min:.2f}, {wden_max:.2f}]")
                    set_axis_limits(ax_left, mid, rng)
                    
                    # Right: π (sampling distribution)
                    if pi_stage2 is not None and pi_stage2.size > 0:
                        ax_right = fig.add_subplot(122, projection="3d")
                        pi_log = np.log10(pi_stage2 + 1e-10)
                        sc_right = ax_right.scatter(pts[:,0], pts[:,1], pts[:,2], c=pi_log, s=ptsize*2,
                                       alpha=0.4, cmap="hot", vmin=pi_log.min(), vmax=pi_log.max())
                        fig.colorbar(sc_right, ax=ax_right, shrink=0.6, label="log₁₀(π)")
                        ax_right.set_title(f"π (Sampling Dist)\np_surf^2.2 × w_den^0.7\nmean={pi_stage2.mean():.3e}")
                        set_axis_limits(ax_right, mid, rng)
                    
                    ax = ax_left  # For compatibility
                else:
                    # Fallback if w_den is None
                    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=ptsize*2, alpha=0.4, c='royalblue')
                    ax.set_title(f"{name}\nN = {len(pts):,} anchor points")
            elif i == 3:
                # Stage 3: Upsampled points + Anchor selection heat map
                anchor_positions = as_numpy(stage.get('anchor_positions'))
                anchor_count = as_numpy(stage.get('anchor_selection_count'))
                
                if anchor_positions is not None and anchor_count is not None:
                    # Create side-by-side visualization
                    plt.close(fig)  # Close single subplot
                    fig = plt.figure(figsize=(18, 7))
                    
                    # Left: Upsampled points
                    ax_left = fig.add_subplot(121, projection="3d")
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc_left = ax_left.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize,
                                             alpha=0.4, cmap="plasma")
                    fig.colorbar(sc_left, ax=ax_left, shrink=0.6, label="Distance from Center")
                    ax_left.set_title(f"Upsampled Points\nN = {len(pts):,}")
                    set_axis_limits(ax_left, mid, rng)
                    
                    # Right: Anchor selection heat map
                    ax_right = fig.add_subplot(122, projection="3d")
                    count_min, count_max = anchor_count.min(), anchor_count.max()
                    count_mean = anchor_count.mean()
                    sc_right = ax_right.scatter(anchor_positions[:,0], anchor_positions[:,1], anchor_positions[:,2], 
                                               c=anchor_count, s=ptsize*3, alpha=0.8,
                                               cmap="hot", vmin=count_min, vmax=count_max)
                    fig.colorbar(sc_right, ax=ax_right, shrink=0.6, label="Selection Count")
                    ax_right.set_title(f"Anchor Selection Heat Map\nN = {len(anchor_positions):,} anchors\n"
                                      f"Count: [{count_min}, {count_max}] (avg: {count_mean:.1f})")
                    set_axis_limits(ax_right, mid, rng)
                    
                    ax = None  # No single ax for this stage
                else:
                    # Fallback: just show upsampled points
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize,
                                   alpha=0.6, cmap="plasma")
                    fig.colorbar(sc, ax=ax, shrink=0.6, label="Distance from Center")
                    ax.set_title(f"{name}\nN = {len(pts):,} points")
            elif i == 4:
                # Stage 4: Z-coordinate (smooth contour-like visualization)
                z_vals = pts[:, 2]  # Z coordinate
                z_min, z_max = z_vals.min(), z_vals.max()
                print(f"    [STAGE4] Z coordinate range: [{z_min:.4f}, {z_max:.4f}]")
                sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=z_vals, s=ptsize,
                               alpha=0.4, cmap="viridis", vmin=z_min, vmax=z_max)
                fig.colorbar(sc, ax=ax, shrink=0.6, label="Z Coordinate (Height)")
                ax.set_title(f"{name}\nN = {len(pts):,} points\nZ: [{z_min:.3f}, {z_max:.3f}]")
            elif i == 5 and 'normals' in stage:
                # Stage 5: Normal smoothing - visualize normal Z-component
                nrm = as_numpy(stage['normals'])
                if nrm is not None and nrm.size > 0:
                    nrm_z = nrm[:, 2]  # Z component of normals
                    nrm_z_min, nrm_z_max = nrm_z.min(), nrm_z.max()
                    print(f"    [STAGE5] Normal Z range: [{nrm_z_min:.4f}, {nrm_z_max:.4f}]")
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=nrm_z, s=ptsize,
                                   alpha=0.4, cmap="RdBu_r", vmin=nrm_z_min, vmax=nrm_z_max)
                    fig.colorbar(sc, ax=ax, shrink=0.6, label="Normal Z-component")
                    ax.set_title(f"{name}\nN = {len(pts):,} points\nNormal Z: [{nrm_z_min:.3f}, {nrm_z_max:.3f}]")
                else:
                    # Fallback
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize,
                                   alpha=0.4, cmap="viridis")
                    fig.colorbar(sc, ax=ax, shrink=0.6, label="Distance")
                    ax.set_title(f"{name}\nN = {len(pts):,} points")
            elif i == 6 and 'cov' in stage:
                # Stage 6: Covariance - visualize covariance scale (trace or determinant)
                cov = as_numpy(stage['cov'])
                if cov is not None and cov.size > 0:
                    # Compute trace (sum of eigenvalues) as a measure of covariance size
                    trace = np.trace(cov, axis1=1, axis2=2)
                    
                    # Smooth visualization: Percentile clipping for outliers (5th~95th)
                    trace_p5, trace_p95 = np.percentile(trace, [5, 95])
                    trace_clipped = np.clip(trace, trace_p5, trace_p95)
                    
                    # Adaptive alpha based on covariance magnitude
                    # Lower trace → higher alpha (more certain → more visible)
                    trace_norm = (trace_clipped - trace_p5) / (trace_p95 - trace_p5 + 1e-12)  # [0,1]
                    alpha_min, alpha_max = 0.15, 0.65  # Range increased from 0.2 to 0.15~0.65
                    alpha_values = alpha_max - trace_norm * (alpha_max - alpha_min)  # Inverted: low trace → high alpha
                    
                    trace_min, trace_max = trace.min(), trace.max()
                    print(f"    [STAGE6] Covariance trace range: [{trace_min:.6f}, {trace_max:.6f}]")
                    print(f"    [STAGE6] Percentile clipping: [{trace_p5:.6f}, {trace_p95:.6f}] (5th~95th)")
                    print(f"    [STAGE6] Adaptive alpha range: [{alpha_min:.2f}, {alpha_max:.2f}]")
                    
                    # New colormap: "viridis" (purple→blue→green→yellow, standard & clean)
                    # Alternatives: "magma" (black→purple→yellow), "plasma" (purple→orange→yellow)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=trace_clipped, s=ptsize*0.6,
                                   alpha=alpha_values, cmap="viridis", vmin=trace_p5, vmax=trace_p95)
                    fig.colorbar(sc, ax=ax, shrink=0.6, label="Cov Trace (adaptive α)")
                    ax.set_title(f"{name}\nN = {len(pts):,} points\nTrace: [{trace_min:.4e}, {trace_max:.4e}]\nα: [{alpha_min:.2f}~{alpha_max:.2f}] (adaptive)", fontsize=9)
                else:
                    # Fallback
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize,
                                   alpha=0.4, cmap="viridis")
                    fig.colorbar(sc, ax=ax, shrink=0.6, label="Distance")
                    ax.set_title(f"{name}\nN = {len(pts):,} points")
            else:
                # Fallback: Color by radial distance
                c = pts.mean(0)
                r = np.linalg.norm(pts - c[None,:], axis=1)
                sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize,
                               alpha=0.6, cmap="viridis")
                fig.colorbar(sc, ax=ax, shrink=0.6, label="Distance from Center")
                ax.set_title(f"{name}\nN = {len(pts):,} points")
            
            # Set axis limits (skip if ax is None, e.g., stage3 side-by-side)
            if ax is not None:
                set_axis_limits(ax, mid, rng)
        
        # Save individual stage (skip STAGE1, already saved as 1a and 1b)
        stage_path = ep_dir / f"STAGE{i}.png"
        fig.tight_layout()
        fig.savefig(stage_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {stage_path.name}")
    
    # Create combined progression visualization
    if n_stages >= 3:
        # Use 1x6 layout for horizontal progression
        ncols = n_stages  # All stages in one row
        nrows = 1
        fig = plt.figure(figsize=(5*ncols, 6))
        
        for i, (stage, name) in enumerate(zip(stages, stage_names), start=1):
            ax = fig.add_subplot(nrows, ncols, i, projection="3d")
            
            pts = as_numpy(stage.get('points'))
            if pts is None or pts.size == 0:
                ax.text2D(0.5, 0.5, "No points", transform=ax.transAxes, ha="center")
            else:
                # Color based on stage type - same as individual visualizations
                if i == 1:
                    # Stage 1: Show all anchors with p_surf_raw color
                    p_surf_raw_prog = as_numpy(stage.get('p_surf_raw'))
                    if p_surf_raw_prog is not None and p_surf_raw_prog.size > 0:
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], 
                                       c=p_surf_raw_prog, s=ptsize*1.2,
                                       alpha=0.7, cmap="coolwarm", vmin=0, vmax=1)
                    else:
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], 
                                       c='steelblue', s=ptsize*1.2, alpha=0.7)
                elif i == 2:
                    # Stage 2: Density weight (w_den)
                    w_den = as_numpy(stage.get('w_den'))
                    if w_den is not None and w_den.size > 0:
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=w_den, s=ptsize*0.8,
                                       alpha=0.4, cmap="viridis", vmin=w_den.min(), vmax=w_den.max())
                    else:
                        # Fallback
                        c = pts.mean(0)
                        r = np.linalg.norm(pts - c[None,:], axis=1)
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize*0.8,
                                       alpha=0.4, cmap="viridis")
                elif i == 3:
                    # Stage 3: Distance from center (sampling result)
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize*0.8,
                                   alpha=0.4, cmap="plasma")
                elif i == 4:
                    # Stage 4: Z-coordinate (smooth contour visualization)
                    z_vals = pts[:, 2]
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=z_vals, s=ptsize*0.8,
                                   alpha=0.4, cmap="viridis", vmin=z_vals.min(), vmax=z_vals.max())
                elif i == 5 and 'normals' in stage:
                    # Stage 5: Normal Z-component
                    nrm = as_numpy(stage['normals'])
                    if nrm is not None and nrm.size > 0:
                        nrm_z = nrm[:, 2]
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=nrm_z, s=ptsize*0.8,
                                       alpha=0.4, cmap="RdBu_r", vmin=nrm_z.min(), vmax=nrm_z.max())
                    else:
                        c = pts.mean(0)
                        r = np.linalg.norm(pts - c[None,:], axis=1)
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize*0.8,
                                       alpha=0.4, cmap="viridis")
                elif i == 6 and 'cov' in stage:
                    # Stage 6: Covariance trace (smooth visualization with adaptive alpha)
                    cov = as_numpy(stage['cov'])
                    if cov is not None and cov.size > 0:
                        trace = np.trace(cov, axis1=1, axis2=2)
                        # Percentile clipping for smooth visualization
                        trace_p5, trace_p95 = np.percentile(trace, [5, 95])
                        trace_clipped = np.clip(trace, trace_p5, trace_p95)
                        
                        # Adaptive alpha based on trace magnitude (inverted: low trace → high alpha)
                        trace_norm = (trace_clipped - trace_p5) / (trace_p95 - trace_p5 + 1e-12)
                        alpha_min, alpha_max = 0.15, 0.65
                        alpha_values = alpha_max - trace_norm * (alpha_max - alpha_min)
                        
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=trace_clipped, s=ptsize*0.5,
                                       alpha=alpha_values, cmap="viridis", vmin=trace_p5, vmax=trace_p95)
                    else:
                        c = pts.mean(0)
                        r = np.linalg.norm(pts - c[None,:], axis=1)
                        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize*0.8,
                                       alpha=0.4, cmap="viridis")
                else:
                    # Fallback: radial distance
                    c = pts.mean(0)
                    r = np.linalg.norm(pts - c[None,:], axis=1)
                    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=r, s=ptsize*0.8,
                                   alpha=0.4, cmap="viridis")
                
                # No title
                set_axis_limits(ax, mid, rng)
        
        # Filename based on episode type
        if episode == -1:
            prog_filename = "target_progression.png"
        else:
            prog_filename = f"ep{episode:03d}_progression.png"
        
        # No suptitle - clean visualization
        fig.tight_layout(h_pad=1.0, w_pad=1.0)
        
        prog_path = ep_dir / prog_filename
        fig.savefig(prog_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {prog_path.name}")
    
    print(f"✓ All stage visualizations saved to {ep_dir}/")
    print(f"  - Individual stages: STAGE1.png ~ STAGE{n_stages}.png")
    if n_stages >= 3:
        print(f"  - Combined view: {prog_filename}")