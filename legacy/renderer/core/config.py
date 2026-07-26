"""Rendering configuration."""

from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuration for rendering operations."""
    
    return_torch: bool = False
    prefer_cov_precomp: bool = True
    render_normal_map: bool = False
    render_depth: bool = True
    auto_alpha: bool = True
    debug: bool = False