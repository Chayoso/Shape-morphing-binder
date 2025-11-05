"""
SDF-based visualization functions.

Author: CHAYO
"""

import torch
import numpy as np
from pathlib import Path


# =============================================================================
# 🔧 공통 축 설정 유틸 (Y축 버그 픽스)
# =============================================================================
def _set_world_limits(ax, bbox_min, bbox_max, verbose=False):
    """3D 축을 실제 월드 bbox에 정확히 맞춤 (Y 최소 버그 픽스)."""
    bmin = np.asarray(bbox_min, dtype=float)
    bmax = np.asarray(bbox_max, dtype=float)
    
    # 축 범위 설정
    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_zlim(bmin[2], bmax[2])
    
    # 축 비율 동일하게
    extents = (bmax - bmin).astype(float)
    extents = np.maximum(extents, 1e-6)
    
    try:
        ax.set_box_aspect(extents)  # Matplotlib 3.3+
    except Exception:
        pass


def _infer_bbox_from_points(points, bbox_min=None, bbox_max=None, padding=0.1):
    """
    bbox가 주어지지 않은 경우 포인트에서 추정.
    
    Args:
        points: (N, 3) numpy array or torch tensor
        bbox_min: (3,) minimum bbox (optional)
        bbox_max: (3,) maximum bbox (optional)
        padding: padding ratio (default: 0.1 = 10%)
    
    Returns:
        bmin, bmax: (3,) numpy arrays
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    if bbox_min is None or bbox_max is None:
        bmin = points.min(axis=0)
        bmax = points.max(axis=0)
        pad = (bmax - bmin) * padding
        bmin = bmin - pad
        bmax = bmax + pad
    else:
        if isinstance(bbox_min, torch.Tensor):
            bmin = bbox_min.detach().cpu().numpy()
        else:
            bmin = np.asarray(bbox_min, dtype=float)
        
        if isinstance(bbox_max, torch.Tensor):
            bmax = bbox_max.detach().cpu().numpy()
        else:
            bmax = np.asarray(bbox_max, dtype=float)
    
    return bmin, bmax


def sanity_check_limits(name, pts, bmin, bmax, verbose=False):
    """
    축-데이터 일관성 체크 (디버그용).
    
    Args:
        name: 체크할 스테이지 이름
        pts: (N, 3) 포인트
        bmin: (3,) bbox minimum
        bmax: (3,) bbox maximum
        verbose: 출력 여부
    """
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    
    eps = 1e-5
    
    # Y축 체크 (버그 검증용)
    if pmin[1] < bmin[1] - eps:
        msg = f"[{name}] Y-min {pmin[1]:.3f} < bbox_min {bmin[1]:.3f}"
        if verbose:
            print(f"⚠️  {msg}")
    
    if pmax[1] > bmax[1] + eps:
        msg = f"[{name}] Y-max {pmax[1]:.3f} > bbox_max {bmax[1]:.3f}"
        if verbose:
            print(f"⚠️  {msg}")
    
    if verbose:
        print(f"[{name}] Point range: Y=[{pmin[1]:.3f}, {pmax[1]:.3f}]")
        print(f"[{name}] BBox range:  Y=[{bmin[1]:.3f}, {bmax[1]:.3f}]")


def _subsample_random(points, cap=100000, seed=0):
    """
    🔧 무작위 균등 샘플링 (섬 아티팩트 방지).
    
    연속 인덱스(stride) 샘플링은 앵커별 묶음 구조에서 뭉텅이만 뽑혀
    시각화가 섬처럼 끊겨 보이는 문제가 있음. 무작위로 해결.
    
    Args:
        points: (N, 3) 포인트 (torch.Tensor or np.ndarray)
        cap: 최대 표시 개수
        seed: 재현성을 위한 시드
    
    Returns:
        idx: 선택된 인덱스 (numpy array)
    """
    if isinstance(points, torch.Tensor):
        M = points.shape[0]
        device = points.device
        if M <= cap:
            return np.arange(M)
        
        # Torch로 무작위 샘플링 후 numpy 반환
        g = torch.Generator(device=device).manual_seed(seed)
        idx = torch.randperm(M, generator=g, device=device)[:cap]
        return idx.cpu().numpy()
    else:
        M = len(points)
        if M <= cap:
            return np.arange(M)
        
        np.random.seed(seed)
        return np.random.choice(M, cap, replace=False)


def _visualize_input_points(save_path, points, bbox_min=None, bbox_max=None, verbose=True):
    """
    STAGE 0 입력 포인트 클라우드 시각화
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping input points visualization")
        return
    
    # Convert to numpy
    pts = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points
    
    # 🔧 공통 헬퍼로 bbox 추정
    bbox_min, bbox_max = _infer_bbox_from_points(pts, bbox_min, bbox_max, padding=0.1)
    
    # 포인트-bbox 일관성 체크 (silent)
    sanity_check_limits("STAGE0_input", pts, bbox_min, bbox_max, verbose=False)
    
    # Create 1x2 figure
    fig = plt.figure(figsize=(14, 7))
    
    # Color by Z coordinate (depth)
    z_normalized = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-8)
    
    # Common settings
    point_size = 1.5
    alpha = 0.7
    
    # 1) 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=z_normalized, cmap='plasma', s=point_size, alpha=alpha,
                      vmin=0, vmax=1, depthshade=True)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Input Point Cloud', fontsize=14, fontweight='bold')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('Depth (Z normalized)', fontsize=10)
    
    # Title
    title = f"STAGE 0 - Input Point Cloud\n" \
            f"(Volumetric points from mesh/simulation)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 2) Statistics panel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    num_points = pts.shape[0]
    
    # Compute statistics
    centroid = pts.mean(axis=0)
    distances = np.linalg.norm(pts - centroid, axis=1)
    
    stats_text = f"""
Input Point Cloud Statistics (STAGE 0)
{'='*40}

Total Points: {num_points:,}

Bounding Box:
  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]
  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]
  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

Point Positions:
  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]
  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]
  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]

Centroid:
  ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})

Distance from Centroid:
  Mean: {distances.mean():.2f}
  Std:  {distances.std():.2f}
  Max:  {distances.max():.2f}

Depth (Z) Distribution:
  Min:    {pts[:, 2].min():.2f}
  Mean:   {pts[:, 2].mean():.2f}
  Max:    {pts[:, 2].max():.2f}
  Std:    {pts[:, 2].std():.2f}
"""
    
    ax2.text(0.1, 0.5, stats_text, 
             fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  [SAVE] Input points visualization: {save_path}")


def _visualize_sampled_points(save_path, points, normals, anchors, bbox_min, bbox_max, verbose=True):
    """
    STAGE 3 샘플링된 포인트 시각화 - Normal Z로 색상
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping sampled points visualization")
        return
    
    # Convert to numpy
    pts = points.detach().cpu().numpy()
    norms = normals.detach().cpu().numpy()
    anchors_np = anchors.detach().cpu().numpy()
    bbox_min = bbox_min.detach().cpu().numpy()
    bbox_max = bbox_max.detach().cpu().numpy()
    
    total_points = pts.shape[0]
    total_anchors = anchors_np.shape[0]
    
    # 🔧 디버그: 포인트-bbox 일관성 체크
    if verbose:
        sanity_check_limits("STAGE3_sampled", pts, bbox_min, bbox_max, verbose=False)
    
    # 🔧 무작위 서브샘플링 (섬 아티팩트 방지)
    cap = 100000
    if pts.shape[0] > cap:
        idx = _subsample_random(pts, cap=cap, seed=3)  # seed=3 for stage 3
        pts = pts[idx]
        norms = norms[idx]
        if verbose:
            print(f"  [viz] Subsampled for display: {len(idx):,} / {total_points:,} points")
    
    # Create 1x2 figure
    fig = plt.figure(figsize=(14, 7))
    
    # Common settings
    point_size = 0.5  # Smaller for dense points
    alpha = 0.4
    
    # Color by normal Z component
    colors_normalZ = norms[:, 2]
    
    # 1) 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=colors_normalZ, cmap='coolwarm', s=point_size, alpha=alpha,
                      vmin=-1, vmax=1)
    ax1.view_init(elev=25, azim=-45)  # 🔥 anchor와 같은 각도
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Sampled Points - Normal Z', fontsize=14, fontweight='bold')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('Normal Z', fontsize=10)
    
    # Title
    title = f"STAGE 3 - Sampled Points\n" \
            f"(Red=Up, Blue=Down)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 2) Statistics panel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    shown_points = pts.shape[0]
    expansion = total_points / total_anchors if total_anchors > 0 else 0
    norm_mag = np.linalg.norm(norms, axis=1)
    
    stats_text = f"""
Sampled Points Statistics (STAGE 3)
{'='*40}

Points: {shown_points:,} shown / {total_points:,} total

Sampling:
  Anchors:  {total_anchors:,}
  Sampled:  {total_points:,}
  Expansion: {expansion:.1f}×

Bounding Box:
  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]
  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]
  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

Point Positions:
  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]
  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]
  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]

Normal Magnitude:
  Mean: {norm_mag.mean():.4f}
  Std:  {norm_mag.std():.4f}

Normal Z Distribution:
  Mean: {norms[:, 2].mean():.4f}
  Up (Z>0):   {(norms[:, 2] > 0).mean()*100:.1f}%
  Down (Z<0): {(norms[:, 2] < 0).mean()*100:.1f}%
"""
    
    ax2.text(0.1, 0.5, stats_text, 
             fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  [SAVE] Sampled points visualization: {save_path}")


def _visualize_anchors(save_path, anchors, normals, p_surf_raw, bbox_min, bbox_max, verbose=True):
    """
    개선된 표면 앵커 시각화 - 앵커 확률(p_surf_raw) 중심
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping anchor visualization")
        return
    
    # Convert to numpy
    pts = anchors.detach().cpu().numpy()
    norms = normals.detach().cpu().numpy()
    p_surf = p_surf_raw.detach().cpu().numpy()
    bbox_min = bbox_min.detach().cpu().numpy()
    bbox_max = bbox_max.detach().cpu().numpy()
    
    # 🔧 디버그: 포인트-bbox 일관성 체크
    if verbose:
        sanity_check_limits("STAGE2_anchors", pts, bbox_min, bbox_max, verbose=False)
    
    # Create 1x2 figure (1 view + statistics)
    fig = plt.figure(figsize=(14, 7))
    
    # 🔥 Normalize p_surf to [0, 1] for colormap
    p_surf_min = p_surf.min()
    p_surf_max = p_surf.max()
    if p_surf_max > p_surf_min:
        p_surf_normalized = (p_surf - p_surf_min) / (p_surf_max - p_surf_min)
    else:
        p_surf_normalized = np.zeros_like(p_surf)
    
    # 🔥 Depth-based coloring (Z 좌표로 depth 표현)
    z_normalized = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-8)
    
    # Common settings - 개선된 시각화
    point_size = 2.5
    alpha = 0.8
    
    # 1) 3D View - 살짝 오른쪽 위로 회전
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=z_normalized, cmap='coolwarm', s=point_size, alpha=alpha, 
                      edgecolors='none', vmin=0, vmax=1, depthshade=True)
    ax1.view_init(elev=25, azim=-45)  # 🔥 살짝 위(25°) + 오른쪽으로
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Surface Anchors (Depth Shading)', fontsize=14, fontweight='bold')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    
    # Grid for better depth perception
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('Depth (Z height)', fontsize=10)
    
    # 전체 제목
    title = f"Surface Anchors Visualization\n" \
            f"(Red=High Z, Blue=Low Z - Depth Shading)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 2) Statistics panel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    num_anchors = pts.shape[0]
    norm_mag = np.linalg.norm(norms, axis=1)
    
    stats_text = f"""
Surface Anchor Statistics (STAGE 2)
{'='*40}

Total Anchors: {num_anchors:,}

Bounding Box:
  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]
  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]
  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

Anchor Positions:
  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]
  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]
  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]

Depth (Z) Distribution:
  Min:    {pts[:, 2].min():.2f}
  Mean:   {pts[:, 2].mean():.2f}
  Max:    {pts[:, 2].max():.2f}
  Std:    {pts[:, 2].std():.2f}
  (Red=High, Blue=Low)

p_surf_raw (band probability):
  Mean:   {p_surf.mean():.6e}
  Median: {np.median(p_surf):.6e}
  Range:  [{p_surf.min():.6e}, {p_surf.max():.6e}]

Normal Magnitude:
  Mean: {norm_mag.mean():.4f}
  Std:  {norm_mag.std():.4f}
  (Should be ~1.0 if normalized)
"""
    
    ax2.text(0.1, 0.5, stats_text, 
             fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  [SAVE] Anchor visualization: {save_path}")


def _visualize_sdf_grid(save_path, sdf_grid, bbox_min, bbox_max, verbose=True):
    """Visualize SDF grid slices with matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping SDF visualization")
        return
    
    R = sdf_grid.shape[0]
    
    # Create 2x2 figure
    fig = plt.figure(figsize=(16, 14))
    
    # XY slice (Z=middle)
    ax1 = fig.add_subplot(2, 2, 1)
    z_mid = R // 2
    z_world = bbox_min[2] + (bbox_max[2] - bbox_min[2]) * (z_mid / (R - 1))
    xy_slice = sdf_grid[:, :, z_mid].T
    extent = [bbox_min[0], bbox_max[0], bbox_min[1], bbox_max[1]]  # [left, right, bottom, top]
    im1 = ax1.imshow(xy_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2, extent=extent, aspect='equal')
    
    # 🔥 사용자 패치 E: meshgrid로 명시적 월드 좌표 제공
    x_coords = np.linspace(bbox_min[0], bbox_max[0], R)
    y_coords = np.linspace(bbox_min[1], bbox_max[1], R)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
    ax1.contour(X, Y, xy_slice, levels=[0.0], colors='yellow', linewidths=2)
    
    ax1.set_title(f'SDF XY Slice (Z={z_world:.2f})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (world)'); ax1.set_ylabel('Y (world)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='SDF value')
    
    # XZ slice (Y=middle)
    ax2 = fig.add_subplot(2, 2, 2)
    y_mid = R // 2
    y_world = bbox_min[1] + (bbox_max[1] - bbox_min[1]) * (y_mid / (R - 1))
    xz_slice = sdf_grid[:, y_mid, :].T
    extent = [bbox_min[0], bbox_max[0], bbox_min[2], bbox_max[2]]
    im2 = ax2.imshow(xz_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2, extent=extent, aspect='equal')
    
    # 🔥 사용자 패치 E: meshgrid로 명시적 월드 좌표 제공
    x_coords = np.linspace(bbox_min[0], bbox_max[0], R)
    z_coords = np.linspace(bbox_min[2], bbox_max[2], R)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='xy')
    ax2.contour(X, Z, xz_slice, levels=[0.0], colors='yellow', linewidths=2)
    
    ax2.set_title(f'SDF XZ Slice (Y={y_world:.2f})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (world)'); ax2.set_ylabel('Z (world)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2, label='SDF value')
    
    # YZ slice (X=middle)
    ax3 = fig.add_subplot(2, 2, 3)
    x_mid = R // 2
    x_world = bbox_min[0] + (bbox_max[0] - bbox_min[0]) * (x_mid / (R - 1))
    yz_slice = sdf_grid[x_mid, :, :].T
    extent = [bbox_min[1], bbox_max[1], bbox_min[2], bbox_max[2]]
    im3 = ax3.imshow(yz_slice, cmap='RdBu', origin='lower', vmin=-0.2, vmax=0.2, extent=extent, aspect='equal')
    
    # 🔥 사용자 패치 E: meshgrid로 명시적 월드 좌표 제공
    y_coords = np.linspace(bbox_min[1], bbox_max[1], R)
    z_coords = np.linspace(bbox_min[2], bbox_max[2], R)
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='xy')
    ax3.contour(Y, Z, yz_slice, levels=[0.0], colors='yellow', linewidths=2)
    
    ax3.set_title(f'SDF YZ Slice (X={x_world:.2f})', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Y (world)'); ax3.set_ylabel('Z (world)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax3, label='SDF value')
    
    # Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    sdf_min, sdf_max = sdf_grid.min(), sdf_grid.max()
    sdf_mean = sdf_grid.mean()
    surface_voxels = (np.abs(sdf_grid) < 0.05).sum()
    total_voxels = R ** 3
    
    # Compute bbox center
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min
    
    stats_text = f"""
SDF Grid Statistics (STAGE 1)
{'='*40}

Grid Size: {R}³ = {total_voxels:,} voxels

Bounding Box:
  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]
  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]
  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

Center: [{bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f}]
Size: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]

SDF Range: [{sdf_min:.4f}, {sdf_max:.4f}]
SDF Mean:  {sdf_mean:.4f}

Surface Voxels (|SDF| < 0.05):
  Count: {surface_voxels:,}
  Ratio: {surface_voxels/total_voxels:.2%}

Note: Object may be off-center if mesh
      is not centered at origin
"""
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  ✅ SDF visualization saved: {save_path}")


def _visualize_projected_points(save_path, points, normals, bbox_min, bbox_max, verbose=True):
    """
    STAGE 4 투영된 포인트 시각화 - φ=0 표면에 투영된 상태
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping projected points visualization")
        return
    
    # Convert to numpy
    pts = points.detach().cpu().numpy()
    norms = normals.detach().cpu().numpy()
    bbox_min = bbox_min.detach().cpu().numpy()
    bbox_max = bbox_max.detach().cpu().numpy()
    
    total_points = pts.shape[0]
    
    # 🔧 디버그: 포인트-bbox 일관성 체크
    if verbose:
        sanity_check_limits("STAGE4_projected", pts, bbox_min, bbox_max, verbose=False)
    
    # 🔧 무작위 서브샘플링 (섬 아티팩트 방지)
    cap = 100000
    if pts.shape[0] > cap:
        idx = _subsample_random(pts, cap=cap, seed=4)  # seed=4 for stage 4
        pts = pts[idx]
        norms = norms[idx]
        if verbose:
            print(f"  [viz] Subsampled for display: {len(idx):,} / {total_points:,} points")
    
    # Create 1x2 figure
    fig = plt.figure(figsize=(14, 7))
    
    # Common settings
    point_size = 0.5
    alpha = 0.5
    
    # Color by Z coordinate (depth)
    z_normalized = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min() + 1e-8)
    
    # 1) 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=z_normalized, cmap='viridis', s=point_size, alpha=alpha,
                      vmin=0, vmax=1, depthshade=True)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Projected Points (φ=0)', fontsize=14, fontweight='bold')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('Depth (Z normalized)', fontsize=10)
    
    # Title
    title = f"STAGE 4 - Projected Points\n" \
            f"(After φ=0 projection + Tangent smoothing)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 2) Statistics panel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    shown_points = pts.shape[0]
    norm_mag = np.linalg.norm(norms, axis=1)
    
    stats_text = f"""
Projected Points Statistics (STAGE 4)
{'='*40}

Points: {shown_points:,} shown / {total_points:,} total

Bounding Box:
  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}]
  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]
  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}]

Point Positions:
  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]
  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]
  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]

Normal Magnitude:
  Mean: {norm_mag.mean():.4f}
  Std:  {norm_mag.std():.4f}
  Min:  {norm_mag.min():.4f}
  Max:  {norm_mag.max():.4f}

Normal Z Component:
  Mean: {norms[:, 2].mean():.4f}
  Up (Z>0):   {(norms[:, 2] > 0).mean()*100:.1f}%
  Down (Z<0): {(norms[:, 2] < 0).mean()*100:.1f}%
"""
    
    ax2.text(0.1, 0.5, stats_text, 
             fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  [SAVE] Projected points visualization: {save_path}")


def _visualize_normals(save_path, points, normals, bbox_min, bbox_max, verbose=True):
    """
    STAGE 5 노멀 재계산 후 시각화 - 노멀 벡터 포함
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping normals visualization")
        return
    
    # Convert to numpy
    pts = points.detach().cpu().numpy()
    norms = normals.detach().cpu().numpy()
    bbox_min = bbox_min.detach().cpu().numpy()
    bbox_max = bbox_max.detach().cpu().numpy()
    
    # Downsample for normal arrows (너무 많으면 보이지 않음)
    if len(pts) > 5000:
        idx = np.random.choice(len(pts), 5000, replace=False)
        pts_vis = pts[idx]
        norms_vis = norms[idx]
    else:
        pts_vis = pts
        norms_vis = norms
    
    # 🔧 디버그: 포인트-bbox 일관성 체크
    if verbose:
        sanity_check_limits("STAGE4_normals", pts, bbox_min, bbox_max, verbose=False)
    
    # 화살표 길이 계산
    size = bbox_max - bbox_min
    arrow_scale = float(size.mean()) * 0.03
    
    # Create 1x2 figure
    fig = plt.figure(figsize=(14, 7))
    
    # Color by normal Z component
    colors_normalZ = norms_vis[:, 2]
    
    # 1) 3D View with normal arrows
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Points
    sc1 = ax1.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2],
                      c=colors_normalZ, cmap='coolwarm', s=1.0, alpha=0.6,
                      vmin=-1, vmax=1)
    
    # Normal arrows (subsample further)
    arrow_step = max(1, len(pts_vis) // 200)  # 최대 200개 화살표
    for i in range(0, len(pts_vis), arrow_step):
        p = pts_vis[i]
        n = norms_vis[i] * arrow_scale
        ax1.quiver(p[0], p[1], p[2], n[0], n[1], n[2],
                  color='red', alpha=0.7, arrow_length_ratio=0.3, linewidth=1.5)
    
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Points + Normals (∇φ)', fontsize=14, fontweight='bold')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    ax1.grid(True, alpha=0.3)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
    cbar.set_label('Normal Z', fontsize=10)
    
    # Title
    title = f"STAGE 5 - Normal Refinement\n" \
            f"(Red arrows = normals, colored by Z component)"
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 2) Statistics panel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    num_points = pts.shape[0]
    norm_mag = np.linalg.norm(norms, axis=1)
    
    # Normal consistency check
    norm_dot = np.abs((norms[:-1] * norms[1:]).sum(axis=1))
    
    stats_text = f"""
Normal Statistics (STAGE 5)
{'='*40}

Total Points: {num_points:,}
Visualized: {len(pts_vis):,}
Arrows shown: ~{len(pts_vis)//arrow_step}

Normal Magnitude:
  Mean: {norm_mag.mean():.6f}
  Std:  {norm_mag.std():.6f}
  Min:  {norm_mag.min():.6f}
  Max:  {norm_mag.max():.6f}
  (Should be ~1.0 if normalized)

Normal Direction:
  X mean: {norms[:, 0].mean():.4f}
  Y mean: {norms[:, 1].mean():.4f}
  Z mean: {norms[:, 2].mean():.4f}

Normal Z Distribution:
  Up (Z>0):   {(norms[:, 2] > 0).mean()*100:.1f}%
  Down (Z<0): {(norms[:, 2] < 0).mean()*100:.1f}%

Normal Consistency:
  Adjacent dot: {norm_dot.mean():.4f}
  (Higher = smoother, max=1.0)
"""
    
    ax2.text(0.1, 0.5, stats_text, 
             fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"  [SAVE] Normals visualization: {save_path}")


def _visualize_curvature_data(save_path, points, planarity, anisotropy, principal_curv=None, bbox_min=None, bbox_max=None, verbose=True):
    """
    STAGE 1.5 곡률 데이터 시각화 (planarity, anisotropy, principal curvatures)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available, skipping curvature visualization")
        return
    
    # Convert to numpy
    pts = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points
    plan = planarity.detach().cpu().numpy() if isinstance(planarity, torch.Tensor) else planarity
    aniso = anisotropy.detach().cpu().numpy() if isinstance(anisotropy, torch.Tensor) else anisotropy
    
    if principal_curv is not None:
        pcurv = principal_curv.detach().cpu().numpy() if isinstance(principal_curv, torch.Tensor) else principal_curv
        k1, k2 = pcurv[:, 0], pcurv[:, 1]
        k_mean = (k1 + k2) / 2.0
        has_pcurv = True
    else:
        has_pcurv = False
    
    # 🔧 공통 헬퍼로 bbox 추정
    bbox_min, bbox_max = _infer_bbox_from_points(pts, bbox_min, bbox_max, padding=0.1)
    
    # 🔧 디버그: 포인트-bbox 일관성 체크
    if verbose:
        sanity_check_limits("STAGE1.5_curvature", pts, bbox_min, bbox_max, verbose=False)
    
    # Downsampling for visualization
    if pts.shape[0] > 50000:
        indices = np.random.choice(pts.shape[0], 50000, replace=False)
        pts = pts[indices]
        plan = plan[indices]
        aniso = aniso[indices]
        if has_pcurv:
            k_mean = k_mean[indices]
    
    # Create figure
    num_plots = 3 if has_pcurv else 2
    fig = plt.figure(figsize=(7 * num_plots, 7))
    
    # Plot 1: Planarity
    ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
    sc1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=plan, cmap='viridis', s=0.5, alpha=0.6)
    ax1.set_title(f'Planarity (s)\nmean={plan.mean():.3f}, std={plan.std():.3f}', fontsize=12)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    _set_world_limits(ax1, bbox_min, bbox_max, verbose=False)
    plt.colorbar(sc1, ax=ax1, shrink=0.5, label='Planarity [0=vol, 1=flat]')
    
    # Plot 2: Anisotropy
    ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
    sc2 = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=aniso, cmap='plasma', s=0.5, alpha=0.6)
    ax2.set_title(f'Anisotropy (ρ)\nmean={aniso.mean():.3f}, std={aniso.std():.3f}', fontsize=12)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    _set_world_limits(ax2, bbox_min, bbox_max, verbose=False)
    plt.colorbar(sc2, ax=ax2, shrink=0.5, label='Anisotropy [0=uniform, 1=sharp]')
    
    # Plot 3: Mean Curvature (if available)
    if has_pcurv:
        ax3 = fig.add_subplot(1, num_plots, 3, projection='3d')
        sc3 = ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=k_mean, cmap='hot', s=0.5, alpha=0.6)
        ax3.set_title(f'Mean Curvature (κ)\nmean={k_mean.mean():.4f}, max={k_mean.max():.4f}', fontsize=12)
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        _set_world_limits(ax3, bbox_min, bbox_max, verbose=False)
        plt.colorbar(sc3, ax=ax3, shrink=0.5, label='Mean Curvature')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"  [SAVE] Curvature data visualization: {save_path}")


__all__ = [
    "_visualize_input_points",
    "_visualize_sampled_points",
    "_visualize_anchors",
    "_visualize_sdf_grid",
    "_visualize_projected_points",
    "_visualize_normals",
    "_visualize_curvature_data",
]
