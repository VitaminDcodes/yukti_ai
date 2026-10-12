"""
paint_simulation/deposition_model.py

Mathematical Spray Painting Deposition Model.
Implements industrial coating physics based on:
- Conical spray footprint with Gaussian radial flux distribution
- Standoff distance inverse-square attenuation
- Surface normal incidence angle factor (Lambert's cosine law)
- Cumulative dry film thickness (DFT) tracking in microns (um)
- Optical coating absorption via the Beer-Lambert law
"""

import numpy as np


class SprayDepositionModel:
    """
    Simulates physical paint deposition from an industrial spray gun.
    """

    def __init__(
        self,
        nominal_standoff: float = 0.10,      # 100 mm close standoff
        spray_cone_angle_deg: float = 22.0,   # Half-angle of spray cone (degrees)
        peak_deposition_rate: float = 150.0,  # um/s at nominal standoff
        nominal_thickness: float = 50.0,      # 50 um target dry film thickness
        flow_rate_ml_min: float = 400.0       # Fluid delivery
    ):
        self.h0 = nominal_standoff
        self.cone_angle_rad = np.radians(spray_cone_angle_deg)
        self.tan_cone = np.tan(self.cone_angle_rad)
        self.r0 = self.h0 * self.tan_cone     # Footprint radius (~40 mm)
        self.peak_rate = peak_deposition_rate # um/s
        self.t_target = nominal_thickness     # um
        self.flow_rate_ml_s = flow_rate_ml_min / 60.0

        # Colors (RGB in [0, 255])
        self.base_color = np.array([245, 245, 245], dtype=float)   # White primer / BIW
        self.paint_color = np.array([225, 20, 20], dtype=float)    # Vibrant Automotive Gloss Red

    def compute_deposition_patch(
        self,
        tcp_world_pos: np.ndarray,
        spray_dir: np.ndarray,
        grid_x: np.ndarray,
        grid_z: np.ndarray,
        panel_y: float,
        dt: float
    ) -> np.ndarray:
        """
        Calculates incremental paint deposition (in microns) across a 2D surface grid (X, Z).
        
        Parameters:
        - tcp_world_pos: [x, y, z] position of spray nozzle tip in world frame
        - spray_dir: 3D unit vector along the spray jet axis
        - grid_x: 2D array of X surface coordinates (m)
        - grid_z: 2D array of Z surface coordinates (m)
        - panel_y: Y coordinate of the car surface plane (m)
        - dt: simulation timestep (s)
        
        Returns:
        - delta_thickness: 2D array of added paint thickness in microns (um)
        """
        # Standoff distance along the spray direction (Y-axis distance to car)
        # tcp is at y_tcp < panel_y; spray points towards +Y
        standoff = abs(panel_y - tcp_world_pos[1])
        if standoff < 0.02 or standoff > 0.40:
            return np.zeros_like(grid_x, dtype=float)

        # Footprint radius at current standoff
        radius = standoff * self.tan_cone
        sigma = radius / 2.5

        # Distance from spray center on the surface
        dx = grid_x - tcp_world_pos[0]
        dz = grid_z - tcp_world_pos[2]
        r_sq = dx * dx + dz * dz

        # Mask within spray footprint
        mask = r_sq < (radius * radius)
        if not np.any(mask):
            return np.zeros_like(grid_x, dtype=float)

        # 1. Gaussian radial flux distribution
        flux_profile = np.exp(-0.5 * r_sq[mask] / (sigma * sigma))

        # 2. Standoff distance scaling factor (inverse square law)
        dist_scale = (self.h0 / standoff) ** 2.0

        # 3. Incidence angle factor: cosine between spray direction and surface normal (0, -1, 0)
        # Surface normal points outward (-Y) towards the robot
        cos_theta = max(0.0, float(spray_dir[1]))  # spray points +Y, normal points -Y

        # Instantaneous deposition rate: dot_T = peak_rate * f(r) * dist_scale * cos_theta
        dep_rate = self.peak_rate * flux_profile * dist_scale * cos_theta

        delta_thickness = np.zeros_like(grid_x, dtype=float)
        delta_thickness[mask] = dep_rate * dt
        return delta_thickness

    def thickness_to_rgb(self, thickness_map: np.ndarray) -> np.ndarray:
        """
        Converts a 2D thickness map (um) into an RGB image [H, W, 3] in [0, 255]
        using the Beer-Lambert optical absorption law.
        """
        # Optical opacity alpha in [0, 1]
        # alpha = 1 - exp(-T / (0.4 * T_target))
        # At 20 um alpha ~ 0.63; at 40 um alpha ~ 0.86; at 50 um alpha ~ 0.92
        char_thickness = 12.0  # um (rapid optical saturation to deep red)
        alpha = 1.0 - np.exp(-thickness_map / char_thickness)
        alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]

        # Blend base white primer with automotive gloss red
        rgb = (1.0 - alpha) * self.base_color + alpha * self.paint_color
        return np.clip(rgb, 0, 255).astype(np.uint8)
