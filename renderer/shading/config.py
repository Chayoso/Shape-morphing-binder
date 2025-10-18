"""Shading configuration."""

from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


DEFAULT_LIGHT_DIRECTION = np.array([0.3, -0.5, 0.8], dtype=np.float32)
DEFAULT_LIGHT_POSITION = np.array([2.5, 2.0, -2.0], dtype=np.float32)
DEFAULT_LIGHT_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)

DEFAULT_LIGHT_INTENSITY = 1.0
DEFAULT_AMBIENT_COEFF = 0.10
DEFAULT_DIFFUSE_COEFF = 0.90
DEFAULT_SPECULAR_COEFF = 0.10
DEFAULT_SHININESS = 32.0
DEFAULT_MIN_AMBIENT = 0.05


@dataclass
class LightConfig:
    """
    Light source configuration.
    
    Attributes:
        type: 'directional' or 'point'
        direction: Direction from light to scene (for directional)
        position: Light position in world space (for point)
        color: RGB light color [0, 1]
        intensity: Light intensity multiplier
        attenuation: [c0, c1, c2] for 1/(c0 + c1*d + c2*d²)
        ambient: Ambient coefficient
        diffuse: Diffuse coefficient
        specular: Specular coefficient
        shininess: Specular exponent (Phong)
        orient: Normal orientation mode ('view' or 'light')
        two_sided: Enable two-sided shading
        min_ambient: Minimum ambient floor
    """
    type: str = 'directional'
    direction: np.ndarray = None
    position: np.ndarray = None
    color: np.ndarray = None
    intensity: float = DEFAULT_LIGHT_INTENSITY
    attenuation: np.ndarray = None
    ambient: float = DEFAULT_AMBIENT_COEFF
    diffuse: float = DEFAULT_DIFFUSE_COEFF
    specular: float = DEFAULT_SPECULAR_COEFF
    shininess: float = DEFAULT_SHININESS
    orient: str = 'view'
    two_sided: bool = True
    min_ambient: float = DEFAULT_MIN_AMBIENT
    
    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.direction is None:
            self.direction = DEFAULT_LIGHT_DIRECTION.copy()
        if self.position is None:
            self.position = DEFAULT_LIGHT_POSITION.copy()
        if self.color is None:
            self.color = DEFAULT_LIGHT_COLOR.copy()
        if self.attenuation is None:
            self.attenuation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> 'LightConfig':
        """Create LightConfig from dictionary."""
        return cls(
            type=cfg.get('type', 'directional'),
            direction=np.asarray(cfg.get('direction', DEFAULT_LIGHT_DIRECTION), dtype=np.float32),
            position=np.asarray(cfg.get('position', DEFAULT_LIGHT_POSITION), dtype=np.float32),
            color=np.asarray(cfg.get('color', DEFAULT_LIGHT_COLOR), dtype=np.float32),
            intensity=float(cfg.get('intensity', DEFAULT_LIGHT_INTENSITY)),
            attenuation=np.asarray(cfg.get('attenuation', [1.0, 0.0, 0.0]), dtype=np.float32),
            ambient=float(cfg.get('ambient', DEFAULT_AMBIENT_COEFF)),
            diffuse=float(cfg.get('diffuse', DEFAULT_DIFFUSE_COEFF)),
            specular=float(cfg.get('specular', DEFAULT_SPECULAR_COEFF)),
            shininess=float(cfg.get('shininess', DEFAULT_SHININESS)),
            orient=cfg.get('orient', 'view'),
            two_sided=bool(cfg.get('two_sided', True)),
            min_ambient=float(cfg.get('min_ambient', DEFAULT_MIN_AMBIENT))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'direction': self.direction.tolist() if isinstance(self.direction, np.ndarray) else self.direction,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'color': self.color.tolist() if isinstance(self.color, np.ndarray) else self.color,
            'intensity': self.intensity,
            'attenuation': self.attenuation.tolist() if isinstance(self.attenuation, np.ndarray) else self.attenuation,
            'ambient': self.ambient,
            'diffuse': self.diffuse,
            'specular': self.specular,
            'shininess': self.shininess,
            'orient': self.orient,
            'two_sided': self.two_sided,
            'min_ambient': self.min_ambient,
        }