"""
paint_simulation/spray_visualizer.py

3D Dynamic Spray Jet Visualizer for MuJoCo.
Renders the physical spray plume and surface impact footprint in the 3D viewer
using user_scn overlay geometries.
"""

import mujoco
import numpy as np


class SprayVisualizer:
    """
    Renders the active spray cone and impingement footprint in MuJoCo viewer's user_scn.
    """

    def __init__(self, cone_angle_deg: float = 20.0):
        self.cone_angle_rad = np.radians(cone_angle_deg)
        self.tan_cone = np.tan(self.cone_angle_rad)
        self.spray_color = np.array([0.90, 0.15, 0.15, 0.35], dtype=np.float32)  # Translucent red
        self.impact_color = np.array([1.00, 0.20, 0.20, 0.60], dtype=np.float32)

    def draw_spray_cone(
        self,
        viewer,
        tcp_pos: np.ndarray,
        spray_dir: np.ndarray,
        standoff_dist: float
    ):
        """
        Clean visualizer: ensures no artificial cylinder or plume geom is rendered
        in front of the paint nozzle, allowing direct dynamic texture painting on the car.
        """
        if viewer is not None and hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
