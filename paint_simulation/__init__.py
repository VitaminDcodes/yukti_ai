"""
paint_simulation package

Comprehensive industrial spray-painting simulation engine for automotive manufacturing.
Provides mathematical paint deposition, dynamic texture coating, and 3D spray visualization.
"""

from .deposition_model import SprayDepositionModel
from .surface_painter import SurfacePainter
from .spray_visualizer import SprayVisualizer
from .report import save_thickness_heatmap

__all__ = [
    "SprayDepositionModel",
    "SurfacePainter",
    "SprayVisualizer",
    "save_thickness_heatmap"
]
