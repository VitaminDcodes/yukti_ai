"""
paint_simulation/surface_painter.py

Real-Time Dynamic Surface Painter for MuJoCo.
Connects the mathematical spray deposition physics engine to the car's 3D texture buffer,
turning the white car body into vivid red in real time as the spray gun deposits paint.
"""

import mujoco
import numpy as np
from .deposition_model import SprayDepositionModel


class SurfacePainter:
    """
    Manages the 2D surface paint texture and film thickness accumulation on the car workpiece.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        texture_name: str = "car_paint_tex",
        res_x: int = 512,
        res_z: int = 512,
        car_x_bounds: tuple[float, float] = (-2.40, 2.40),
        car_z_bounds: tuple[float, float] = (0.02, 1.62),
        panel_y: float = -0.95
    ):
        self.model = model
        self.res_x = res_x
        self.res_z = res_z
        self.x_min, self.x_max = car_x_bounds
        self.z_min, self.z_max = car_z_bounds
        self.panel_y = panel_y

        # Initialize deposition physics model
        self.physics = SprayDepositionModel()

        # Locate texture in MuJoCo model
        self.tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name)
        if self.tex_id < 0:
            raise ValueError(f"Texture '{texture_name}' not found in MuJoCo model.")

        self.tex_adr = model.tex_adr[self.tex_id]
        self.tex_width = model.tex_width[self.tex_id]
        self.tex_height = model.tex_height[self.tex_id]

        # Accumulated Dry Film Thickness (DFT) map in microns (um)
        # Shape: [res_z, res_x] (row=Z, col=X)
        self.thickness_map = np.zeros((self.res_z, self.res_x), dtype=float)

        # Coordinate meshgrid for 2D surface evaluation (m)
        x_coords = np.linspace(self.x_min, self.x_max, self.res_x)
        z_coords = np.linspace(self.z_min, self.z_max, self.res_z)
        self.grid_x, self.grid_z = np.meshgrid(x_coords, z_coords)

        # Initialize texture buffer to primer white
        self._init_texture_buffer()

        # Metrics tracking
        self.total_paint_sprayed_ml = 0.0

    def _init_texture_buffer(self):
        """Initializes MuJoCo texture buffer to primer white [245, 245, 245]."""
        total_bytes = self.tex_height * self.tex_width * 3
        tex_buf = self.model.tex_data[self.tex_adr : self.tex_adr + total_bytes].reshape(
            self.tex_height, self.tex_width, 3
        )
        tex_buf[:] = self.physics.base_color.astype(np.uint8)

    def update(self, tcp_world_pos: np.ndarray, spray_dir: np.ndarray, dt: float, viewer=None):
        """
        Computes physical paint deposition at current TCP pose, updates film thickness,
        writes into the MuJoCo texture buffer, and triggers live GPU texture re-upload.
        """
        # Standoff check relative to car outer panel
        standoff = abs(self.panel_y - tcp_world_pos[1])
        if standoff < 0.02 or standoff > 0.45:
            return

        # Track total paint volume consumed
        self.total_paint_sprayed_ml += self.physics.flow_rate_ml_s * dt

        # Spray footprint radius on surface (cone expansion)
        radius = max(0.06, standoff * self.physics.tan_cone * 2.2)
        x_c, z_c = tcp_world_pos[0], tcp_world_pos[2]

        # Horizontal column indices: X maps from x_min (col 0) to x_max (col W-1)
        c_min = int(np.clip((x_c - radius - self.x_min) / (self.x_max - self.x_min) * (self.res_x - 1), 0, self.res_x - 1))
        c_max = int(np.clip((x_c + radius - self.x_min) / (self.x_max - self.x_min) * (self.res_x - 1) + 1, 0, self.res_x))

        # Vertical row indices: V is directly proportional to World Z in the mesh
        # Low Z (rocker/ground) maps to row 0, High Z (roof/cowl) maps to row H-1
        r_min = int(np.clip((z_c - radius - self.z_min) / (self.z_max - self.z_min) * (self.res_z - 1), 0, self.res_z - 1))
        r_max = int(np.clip((z_c + radius - self.z_min) / (self.z_max - self.z_min) * (self.res_z - 1) + 1, 0, self.res_z))

        if c_max <= c_min or r_max <= r_min:
            return

        # Localized coordinate subgrid on car surface
        sub_cols = np.arange(c_min, c_max)
        sub_rows = np.arange(r_min, r_max)
        col_grid, row_grid = np.meshgrid(sub_cols, sub_rows)

        sub_x = self.x_min + col_grid / (self.res_x - 1) * (self.x_max - self.x_min)
        sub_z = self.z_min + row_grid / (self.res_z - 1) * (self.z_max - self.z_min)

        dx = sub_x - x_c
        dz = sub_z - z_c
        r_sq = dx * dx + dz * dz
        mask = r_sq < (radius * radius)

        if not np.any(mask):
            return

        # Physical deposition: Gaussian radial profile + distance scaling + incidence angle
        sigma = radius / 2.2
        flux_profile = np.exp(-0.5 * r_sq[mask] / (sigma * sigma))
        dist_scale = (self.physics.h0 / max(0.04, standoff)) ** 1.5
        cos_theta = max(0.2, float(spray_dir[1]))  # spray points +Y towards car

        delta_T = np.zeros_like(r_sq, dtype=float)
        delta_T[mask] = self.physics.peak_rate * flux_profile * dist_scale * cos_theta * dt

        # Accumulate Dry Film Thickness (DFT)
        self.thickness_map[r_min:r_max, c_min:c_max] += delta_T

        # Convert thickness to vibrant red RGB via Beer-Lambert absorption law
        updated_patch_T = self.thickness_map[r_min:r_max, c_min:c_max]
        patch_rgb = self.physics.thickness_to_rgb(updated_patch_T)

        # Write directly into MuJoCo model.tex_data buffer (left half: robot-side canvas)
        total_bytes = self.tex_height * self.tex_width * 3
        tex_buf = self.model.tex_data[self.tex_adr : self.tex_adr + total_bytes].reshape(
            self.tex_height, self.tex_width, 3
        )
        tex_buf[r_min:r_max, c_min:c_max] = patch_rgb

        # Trigger live GPU texture re-upload in the MuJoCo viewer
        if viewer is not None and hasattr(viewer, 'update_texture'):
            viewer.update_texture(self.tex_id)

    def get_metrics(self) -> dict:
        """
        Computes statistical industrial coating metrics:
        - Target coverage area and percentage
        - Dry film thickness (DFT) mean, max, and uniformity
        """
        painted_mask = self.thickness_map > 1.0  # At least 1 micron deposited
        target_met_mask = self.thickness_map >= (0.7 * self.physics.t_target)  # >= 35 um

        total_cells = float(np.sum(self.thickness_map > 0) + 1e-6)
        covered_cells = float(np.sum(painted_mask))
        target_cells = float(np.sum(target_met_mask))

        # Panel side target area: ~4.8m x 0.8m = 3.84 m^2
        dx = (self.x_max - self.x_min) / self.res_x
        dz = (self.z_max - self.z_min) / self.res_z
        cell_area = dx * dz

        covered_area_m2 = covered_cells * cell_area
        mean_thickness_um = float(np.mean(self.thickness_map[painted_mask])) if covered_cells > 0 else 0.0
        max_thickness_um = float(np.max(self.thickness_map))
        std_thickness_um = float(np.std(self.thickness_map[painted_mask])) if covered_cells > 0 else 0.0

        # Uniformity coefficient: 1 - (std / mean)
        uniformity = max(0.0, 1.0 - (std_thickness_um / (mean_thickness_um + 1e-6)))

        return {
            "mean_thickness_um": mean_thickness_um,
            "max_thickness_um": max_thickness_um,
            "std_thickness_um": std_thickness_um,
            "uniformity_pct": uniformity * 100.0,
            "covered_area_m2": covered_area_m2,
            "target_coverage_pct": (target_cells / (covered_cells + 1e-6)) * 100.0 if covered_cells > 0 else 0.0,
            "paint_sprayed_ml": self.total_paint_sprayed_ml
        }
