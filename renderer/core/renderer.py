"""
Main 3D Gaussian Splatting renderer.

This module provides the GSRenderer3DGS class, a high-level wrapper around
diff-gaussian-rasterization with full gradient flow support for E2E training.
"""

from __future__ import annotations
from typing import Dict, Optional, Union, List, Any, Tuple
from pathlib import Path
import os
import sys
import numpy as np

from ..utils.conversion import to_torch_tensor, to_numpy_array
from ..utils.covariance import (
    pack_covariance_3x3_to_6d,
    unpack_covariance_6d_to_3x3,
    pack_covariance_torch,
    decompose_covariance_to_scale_rotation,
)
from ..utils.projection_2d import project_points_to_screen
from ..utils.debug import debug_print, is_debug_enabled, get_tensor_stats, debug_tensor_info


# ============================================================================
# Rasterizer Import with Path Discovery
# ============================================================================

def _discover_and_import_rasterizer():
    """
    Discover and import diff-gaussian-rasterization library.
    
    Tries multiple strategies:
    1. Direct import (if already in sys.path)
    2. Search common submodule locations
    3. Search parent directories
    
    Returns:
        Tuple of (torch, GaussianRasterizationSettings, GaussianRasterizer)
    
    Raises:
        RuntimeError: If rasterizer cannot be imported
    """
    # Strategy 1: Direct import
    try:
        import torch
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        return torch, GaussianRasterizationSettings, GaussianRasterizer
    except ImportError:
        pass
    
    # Strategy 2: Search common locations
    here = Path(__file__).parent
    repo_root = here.parent.parent
    
    search_paths = [
        repo_root / "submodules" / "diff-gaussian-rasterization",
        repo_root / "diff-gaussian-rasterization",
        repo_root / "gaussian-splatting",
        repo_root / "external" / "diff-gaussian-rasterization",
    ]
    
    for search_path in search_paths:
        if search_path.is_dir() and str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))
    
    # Strategy 3: Try import again
    try:
        import torch
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
        return torch, GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as final_error:
        raise RuntimeError(
            "Failed to import diff-gaussian-rasterization. "
            "Please ensure it's installed or added to sys.path.\n"
            f"Searched paths: {[str(p) for p in search_paths]}\n"
            f"Error: {final_error}"
        ) from final_error


# Import rasterizer
torch, GaussianRasterizationSettings, GaussianRasterizer = _discover_and_import_rasterizer()


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BACKGROUND_COLOR = (1.0, 1.0, 1.0)
DEFAULT_SPLAT_COLOR = (0.7, 0.7, 0.7)
DEFAULT_OPACITY = 1.0


# ============================================================================
# Output Parsing Utilities
# ============================================================================

def parse_color_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Parse color tensor to (H, W, 3) NumPy array.
    
    Args:
        tensor: Color tensor in (C, H, W) or (H, W, C) format
    
    Returns:
        (H, W, 3) NumPy array in [0, 1]
    """
    # Handle channel-first (C, H, W)
    if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
        tensor = tensor.clamp(0, 1).permute(1, 2, 0)
    
    # Handle channel-last (H, W, C)
    elif tensor.ndim == 3 and tensor.shape[-1] in (3, 4):
        tensor = tensor.clamp(0, 1)
    
    else:
        raise ValueError(f"Unexpected color tensor shape: {tensor.shape}")
    
    # Drop alpha channel if present
    if tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    
    return tensor.detach().cpu().numpy()


def parse_2d_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Parse 2D tensor (depth/alpha) to (H, W) NumPy array.
    
    Args:
        tensor: 2D tensor in various formats
    
    Returns:
        (H, W) NumPy array
    """
    # Handle (1, H, W)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Handle (H, W, 1)
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor after squeeze, got {tensor.shape}")
    
    return tensor.detach().cpu().float().numpy()


def normalize_alpha_tensor(alpha: torch.Tensor) -> torch.Tensor:
    """
    Normalize alpha tensor to (H, W) format.
    
    Args:
        alpha: Alpha tensor in various formats
    
    Returns:
        (H, W) alpha tensor
    
    Notes:
        - Handles CHW, HWC, and unusual formats
        - Applies multiple strategies for robustness
    """
    original_shape = alpha.shape
    debug_print(f"[Alpha] Input shape: {original_shape}")
    
    # Strategy 1: Handle 3D tensors
    if alpha.ndim == 3:
        # CHW format
        if alpha.shape[0] in (1, 3, 4):
            if alpha.shape[0] == 1:
                alpha = alpha.squeeze(0)
            else:
                alpha = alpha[0]  # Take first channel
            debug_print(f"[Alpha] CHW → {alpha.shape}")
        
        # HWC format
        elif alpha.shape[-1] in (1, 3, 4):
            if alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)
            else:
                alpha = alpha[..., 0]  # Take first channel
            debug_print(f"[Alpha] HWC → {alpha.shape}")
        
        # Unusual format: use largest dimensions as H, W
        else:
            sizes = list(alpha.shape)
            sorted_dims = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)
            h_idx, w_idx = sorted_dims[0][0], sorted_dims[1][0]
            
            # Remove smallest dimension
            for i in range(3):
                if i not in (h_idx, w_idx):
                    if alpha.shape[i] == 1:
                        alpha = alpha.squeeze(i)
                    else:
                        alpha = alpha.mean(dim=i)
                    break
            debug_print(f"[Alpha] Unusual → {alpha.shape}")
    
    # Strategy 2: Handle 4D tensors (batch dimension)
    elif alpha.ndim == 4:
        alpha = alpha.squeeze(0) if alpha.shape[0] == 1 else alpha[0]
        
        # Recursively handle remaining dimensions
        if alpha.ndim == 3:
            if alpha.shape[0] in (1, 3, 4):
                alpha = alpha[0]
            elif alpha.shape[-1] in (1, 3, 4):
                alpha = alpha[..., 0]
        
        debug_print(f"[Alpha] 4D → {alpha.shape}")
    
    # Strategy 3: Emergency squeeze
    if alpha.ndim != 2:
        debug_print(f"[Alpha] Emergency squeeze from {alpha.shape}")
        
        while alpha.ndim > 2:
            # Remove singleton dimensions first
            if 1 in alpha.shape:
                for i, s in enumerate(alpha.shape):
                    if s == 1:
                        alpha = alpha.squeeze(i)
                        break
            else:
                # Average over first dimension as last resort
                alpha = alpha.mean(dim=0)
        
        debug_print(f"[Alpha] After emergency: {alpha.shape}")
    
    # Final validation
    if alpha.ndim != 2:
        raise AssertionError(
            f"Alpha normalization failed! "
            f"Original: {original_shape}, Final: {alpha.shape}"
        )
    
    debug_print(f"[Alpha] ✅ Final shape: {alpha.shape}")
    return alpha


def synthesize_alpha_from_luminance(image: torch.Tensor) -> torch.Tensor:
    """
    Synthesize alpha channel from RGB luminance.
    
    Args:
        image: (H, W, 3) RGB image
    
    Returns:
        (H, W) alpha channel
    """
    lum = (
        0.2126 * image[..., 0] +
        0.7152 * image[..., 1] +
        0.0722 * image[..., 2]
    )
    return torch.clamp(lum, 0.0, 1.0)


# ============================================================================
# Main Renderer Class
# ============================================================================

class GSRenderer3DGS:
    """
    3D Gaussian Splatting Renderer.
    
    Wrapper around diff-gaussian-rasterization with:
    - Full gradient flow support for E2E training
    - Automatic format conversion
    - Robust fallback mechanisms
    - Normal map rendering
    
    Attributes:
        width: Image width
        height: Image height
        device: Compute device ('cuda' or 'cpu')
        settings: Rasterization settings
        rasterizer: Underlying rasterizer instance
    
    Example:
        >>> renderer = GSRenderer3DGS(
        ...     width=1280, height=720,
        ...     tanfovx=0.5, tanfovy=0.28,
        ...     viewmatrix=view, projmatrix=proj,
        ...     campos=cam_pos
        ... )
        >>> output = renderer.render(xyz, cov, rgb, return_torch=True)
        >>> loss = criterion(output['image'], target)
        >>> loss.backward()  # Gradients flow to xyz, cov, rgb
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        tanfovx: float,
        tanfovy: float,
        viewmatrix: np.ndarray,
        projmatrix: np.ndarray,
        campos: np.ndarray,
        bg: tuple = DEFAULT_BACKGROUND_COLOR,
        sh_degree: int = 0,
        scale_modifier: float = 1.0,
        prefiltered: bool = False,
        debug: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize renderer.
        
        Args:
            width, height: Image dimensions
            tanfovx, tanfovy: Tangent of half FOV
            viewmatrix: (4, 4) world-to-view matrix (transposed)
            projmatrix: (4, 4) projection matrix (transposed)
            campos: (3,) camera position
            bg: Background color (R, G, B)
            sh_degree: Spherical harmonics degree (0 for RGB-only)
            scale_modifier: Global scale multiplier
            prefiltered: Use prefiltered splatting
            debug: Enable debug mode
            device: Compute device
        """
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self._proj_matrix_np = projmatrix.astype(np.float32).copy()
        
        # Create rasterization settings
        self.settings = GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=float(tanfovx),
            tanfovy=float(tanfovy),
            bg=to_torch_tensor(np.array(bg, dtype=np.float32), device=device),
            scale_modifier=float(scale_modifier),
            viewmatrix=to_torch_tensor(viewmatrix, device=device),
            projmatrix=to_torch_tensor(projmatrix, device=device),
            sh_degree=int(sh_degree),
            campos=to_torch_tensor(campos, device=device),
            prefiltered=bool(prefiltered),
            debug=bool(debug)
        )
        
        # Create rasterizer
        self.rasterizer = GaussianRasterizer(self.settings)
        
        debug_print(f"[Renderer] Initialized {width}x{height} on {device}")
    
    def render(
        self,
        xyz: Union[np.ndarray, torch.Tensor],
        cov: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
        rgb: Optional[Union[np.ndarray, torch.Tensor]] = None,
        opacity: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prefer_cov_precomp: bool = True,
        return_torch: bool = False,
        render_normal_map: bool = False
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Render one frame from Gaussians.
        
        Args:
            xyz: (N, 3) means in world space
            cov: (N, 3, 3) or (N, 6) covariances
            rgb: (N, 3) colors [0, 1]
            opacity: (N, 1) opacities [0, 1]
            normals: (N, 3) normals (for normal map rendering)
            prefer_cov_precomp: Use precomputed covariances if possible
            return_torch: Return torch tensors with gradients
            render_normal_map: Also render normal map
        
        Returns:
            Dictionary with keys:
                - 'image': (H, W, 3) RGB
                - 'depth': (H, W) depth map
                - 'alpha': (H, W) alpha channel
                - 'normal_map': (H, W, 3) normal map (if requested)
        
        Notes:
            - If return_torch=True, gradients flow to all inputs
            - Automatic format conversion for mixed NumPy/PyTorch inputs
            - Fallback to scale+rotation if covariance fails
        """
        # Determine if inputs are torch tensors
        is_torch_input = torch.is_tensor(xyz)
        
        # Convert means to appropriate format
        if return_torch and is_torch_input:
            means3D = xyz.to(self.device) if xyz.device.type != self.device else xyz
            xyz_np = xyz.detach().cpu().numpy()
            debug_print("[Renderer] ✅ Gradient flow ENABLED")
        else:
            xyz_np = to_numpy_array(xyz) if not is_torch_input else xyz.detach().cpu().numpy()
            means3D = to_torch_tensor(xyz_np, device=self.device)
            
            if return_torch:
                debug_print("[Renderer] ⚠️ return_torch=True but input is NumPy")
        
        # Project to 2D
        means2D_np, valid = project_points_to_screen(
            xyz_np, self._proj_matrix_np, self.width, self.height
        )
        means2D = to_torch_tensor(means2D_np, device=self.device)
        
        # Handle colors and opacities
        if rgb is None or opacity is None:
            N = len(xyz_np)
            default_rgb = np.full((N, 3), DEFAULT_SPLAT_COLOR, dtype=np.float32)
            default_opacity = np.full((N, 1), DEFAULT_OPACITY, dtype=np.float32)
            
            if rgb is None:
                rgb = default_rgb
            if opacity is None:
                opacity = default_opacity
        
        # Convert to torch
        if return_torch and torch.is_tensor(rgb):
            colors_t = rgb.to(self.device) if rgb.device.type != self.device else rgb
        else:
            colors_t = to_torch_tensor(rgb, device=self.device)
        
        if return_torch and torch.is_tensor(opacity):
            opac_t = opacity.to(self.device) if opacity.device.type != self.device else opacity
        else:
            opac_t = to_torch_tensor(opacity, device=self.device)
        
        # Render main image
        output = self._render_with_covariance(
            means3D, means2D, colors_t, opac_t, cov,
            prefer_cov_precomp, return_torch
        )
        
        # Optionally render normal map
        normal_map_output = None
        if render_normal_map and normals is not None:
            normal_map_output = self._render_normal_map(
                means3D, means2D, opac_t, cov, normals, return_torch
            )
        
        # Parse outputs
        if return_torch:
            return self._parse_torch_outputs(output, normal_map_output)
        else:
            return self._parse_numpy_outputs(output, normal_map_output)
    
    def _render_with_covariance(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
        prefer_cov_precomp: bool,
        return_torch: bool
    ) -> Any:
        """Render using covariance parameterization."""
        output = None
        
        # Try covariance precomputation first
        if prefer_cov_precomp:
            try:
                output = self._try_render_cov_precomp(
                    means3D, means2D, colors, opacities, cov, return_torch
                )
            except Exception as e:
                debug_print(f"[Renderer] cov3D_precomp failed: {e}")
        
        # Fallback to scale+rotation
        if output is None:
            output = self._render_scale_rotation(
                means3D, means2D, colors, opacities, cov
            )
        
        return output
    
    def _try_render_cov_precomp(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
        return_torch: bool
    ) -> Any:
        """Try rendering with precomputed covariance."""
        cov_is_torch = torch.is_tensor(cov) or (
            isinstance(cov, list) and len(cov) > 0 and torch.is_tensor(cov[0])
        )
        
        # Convert covariance to packed format
        if return_torch and cov_is_torch:
            if isinstance(cov, list):
                cov_stacked = torch.stack(cov, dim=0).to(self.device)
                cov_packed = pack_covariance_torch(cov_stacked)
            elif cov.shape[-1] == 6:
                cov_packed = cov.to(self.device)
            else:
                cov_packed = pack_covariance_torch(cov.to(self.device))
            debug_print("[Renderer] Using cov3D_precomp (torch)")
        else:
            cov_np = to_numpy_array(cov) if not isinstance(cov, list) else np.stack([to_numpy_array(c) for c in cov], axis=0)
            cov_packed = to_torch_tensor(pack_covariance_3x3_to_6d(cov_np), device=self.device)
            debug_print("[Renderer] Using cov3D_precomp (numpy)")
        
        # Render
        return self.rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            colors_precomp=colors,
            cov3D_precomp=cov_packed
        )
    
    def _render_scale_rotation(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]
    ) -> Any:
        """Render using scale+rotation fallback."""
        # Convert to numpy
        cov_np = to_numpy_array(cov) if not isinstance(cov, list) else np.stack([to_numpy_array(c) for c in cov], axis=0)
        
        # Unpack if needed
        if cov_np.ndim == 2 and cov_np.shape[1] == 6:
            cov_np = unpack_covariance_6d_to_3x3(cov_np)
        
        # Decompose
        scales, quaternions = decompose_covariance_to_scale_rotation(cov_np)
        
        scales_t = to_torch_tensor(scales, device=self.device)
        rots_t = to_torch_tensor(quaternions, device=self.device)
        
        debug_print("[Renderer] Using scales+rotations fallback")
        
        return self.rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            colors_precomp=colors,
            scales=scales_t,
            rotations=rots_t
        )
    
    def _render_normal_map(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        cov: Union[np.ndarray, torch.Tensor, List[torch.Tensor]],
        normals: Union[np.ndarray, torch.Tensor],
        return_torch: bool
    ) -> Optional[Any]:
        """Render normal map."""
        try:
            # Convert normals to RGB [0, 1]
            if torch.is_tensor(normals):
                normals_rgb = (normals + 1.0) * 0.5
                normals_rgb = torch.clamp(normals_rgb, 0.0, 1.0).to(self.device)
            else:
                normals_np = to_numpy_array(normals)
                normals_rgb_np = np.clip((normals_np + 1.0) * 0.5, 0.0, 1.0)
                normals_rgb = to_torch_tensor(normals_rgb_np, device=self.device)
            
            # Decompose covariance for rendering
            cov_np = to_numpy_array(cov) if not isinstance(cov, list) else np.stack([to_numpy_array(c) for c in cov], axis=0)
            
            if cov_np.ndim == 2 and cov_np.shape[1] == 6:
                cov_np = unpack_covariance_6d_to_3x3(cov_np)
            
            scales, quaternions = decompose_covariance_to_scale_rotation(cov_np)
            
            scales_t = to_torch_tensor(scales, device=self.device)
            rots_t = to_torch_tensor(quaternions, device=self.device)
            
            return self.rasterizer(
                means3D=means3D,
                means2D=means2D,
                opacities=opacities,
                colors_precomp=normals_rgb,
                scales=scales_t,
                rotations=rots_t
            )
        
        except Exception as e:
            debug_print(f"[Renderer] Normal map rendering failed: {e}")
            return None
    
    def _parse_torch_outputs(
        self,
        output: Any,
        normal_map_output: Optional[Any]
    ) -> Dict[str, torch.Tensor]:
        """Parse rasterizer outputs to torch tensors."""
        # Extract main outputs
        if isinstance(output, (list, tuple)):
            color_t = output[0] if len(output) > 0 else None
            depth_t = output[1] if len(output) > 1 else None
            alpha_t = output[2] if len(output) > 2 else None
        else:
            color_t = output
            depth_t = None
            alpha_t = None
        
        # Parse image
        if color_t is not None and color_t.ndim == 3:
            if color_t.shape[0] in (3, 4):
                image_t = color_t.permute(1, 2, 0)[:, :, :3]
            else:
                image_t = color_t[:, :, :3]
        else:
            image_t = color_t
        
        # Parse depth
        if depth_t is not None:
            # Remove singleton dimensions
            while depth_t.ndim > 2 and (depth_t.shape[0] == 1 or depth_t.shape[-1] == 1):
                if depth_t.shape[0] == 1:
                    depth_t = depth_t.squeeze(0)
                elif depth_t.shape[-1] == 1:
                    depth_t = depth_t.squeeze(-1)
        
        # Parse alpha
        if alpha_t is not None:
            alpha_t = normalize_alpha_tensor(alpha_t)
        elif image_t is not None:
            alpha_t = synthesize_alpha_from_luminance(image_t)
            debug_print("[Renderer] Alpha synthesized from luminance")
        
        # Parse normal map
        normal_map_t = None
        if normal_map_output is not None:
            if isinstance(normal_map_output, (list, tuple)):
                nrm_t = normal_map_output[0] if len(normal_map_output) > 0 else None
            else:
                nrm_t = normal_map_output
            
            if nrm_t is not None and nrm_t.ndim == 3:
                if nrm_t.shape[0] in (3, 4):
                    normal_map_t = nrm_t.permute(1, 2, 0)[:, :, :3]
                else:
                    normal_map_t = nrm_t[:, :, :3]
        
        # Debug gradient flow
        if image_t is not None and hasattr(image_t, 'grad_fn'):
            if image_t.grad_fn is not None:
                debug_print(f"[Renderer] ✅ Output has grad_fn: {image_t.grad_fn}")
            else:
                debug_print("[Renderer] ⚠️ Output tensor but no grad_fn")
        
        return {
            'image': image_t,
            'depth': depth_t,
            'alpha': alpha_t,
            'normal_map': normal_map_t,
        }
    
    def _parse_numpy_outputs(
        self,
        output: Any,
        normal_map_output: Optional[Any]
    ) -> Dict[str, np.ndarray]:
        """Parse rasterizer outputs to numpy arrays."""
        # Parse main outputs (legacy parsing logic)
        rgb_np, depth_np, alpha_np = self._legacy_parse_outputs(output)
        
        # Parse normal map
        normal_map_np = None
        if normal_map_output is not None:
            normal_map_np, _, _ = self._legacy_parse_outputs(normal_map_output)
        
        return {
            'image': rgb_np,
            'depth': depth_np,
            'alpha': alpha_np,
            'normal_map': normal_map_np,
        }
    
    def _legacy_parse_outputs(
        self,
        output: Any
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Legacy output parsing (maintains backward compatibility)."""
        # Convert to list
        if isinstance(output, (list, tuple)):
            vals = list(output)
        else:
            vals = [output]
        
        debug_print(f"[Renderer] Output tuple length: {len(vals)}")
        
        # Find color tensor
        color_t = None
        for v in vals:
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                if v.shape[0] in (3, 4) or v.shape[-1] in (3, 4):
                    color_t = v
                    break
        
        if color_t is None:
            raise RuntimeError("Rasterizer did not return a color tensor")
        
        rgb = parse_color_tensor(color_t)
        
        # Extract alpha from RGBA if present
        alpha = None
        if color_t.ndim == 3 and (color_t.shape[0] == 4 or color_t.shape[-1] == 4):
            if color_t.shape[0] == 4:
                alpha = parse_2d_tensor(color_t[3:4, ...])
            else:
                alpha = parse_2d_tensor(color_t[..., 3:4])
            debug_print("[Renderer] Alpha extracted from RGBA")
        
        # Find 2D tensors (depth/alpha candidates)
        twoD_tensors = []
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                if v.ndim == 2 or (v.ndim == 3 and (v.shape[0] == 1 or v.shape[-1] == 1)):
                    twoD_tensors.append((i, v))
        
        # Identify depth and alpha
        depth = None
        alpha_from_map = None
        depth_scores = []
        
        for i, v in twoD_tensors:
            mn, mx, me = get_tensor_stats(v)
            
            # Alpha candidate: values in [0, 1]
            if alpha is None and 0.0 <= mn and mx <= 1.0 and (mx - mn) > 1e-3:
                alpha_from_map = parse_2d_tensor(v)
                debug_print(f"[Renderer] Alpha candidate at index {i}")
            
            # Depth candidate: wider range
            score = -(mx - mn)
            depth_scores.append((score, i))
        
        # Select depth (widest range)
        if depth_scores:
            depth_idx = sorted(depth_scores)[0][1]
            v = [v for (i, v) in twoD_tensors if i == depth_idx][0]
            depth = parse_2d_tensor(v)
            debug_print(f"[Renderer] Depth selected at index {depth_idx}")
        
        # Fallback alpha
        if alpha is None:
            alpha = alpha_from_map
        
        if alpha is None:
            lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            alpha = np.clip(lum, 0.0, 1.0).astype(np.float32)
            debug_print("[Renderer] Alpha synthesized from luminance")
        
        return (
            rgb.astype(np.float32),
            depth.astype(np.float32) if depth is not None else None,
            alpha.astype(np.float32) if alpha is not None else None,
        )