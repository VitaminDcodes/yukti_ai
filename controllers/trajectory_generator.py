"""
controllers/trajectory_generator.py

Parametric industrial surface-normal trajectory generator for automotive painting cell.
Generates smooth 3D tool center point (TCP) waypoints and orientation matrices
maintaining constant standoff distance (250mm) and dynamic normal pitch compensation.
"""

import numpy as np


def get_industrial_curved_waypoint(t: float, x_speed: float = 0.08, z_freq: float = 0.25):
    """
    Generates target 3D TCP position and orientation matching local car surface curvature.
    
    Parameters:
    - t: Current simulation time (seconds)
    - x_speed: Forward travel speed along the car length (m/s)
    - z_freq: Vertical raster frequency (Hz)
    
    Returns:
    - target_pos: np.ndarray [x, y, z] target TCP position in world coordinates
    - rot_matrix: np.ndarray (3, 3) orientation matrix keeping spray nozzle normal to panel
    """
    # 1. Longitudinal rail motion along car (car spans -2.4m to +2.4m)
    # Starts at -2.2m and sweeps forward to +2.2m
    x = -2.2 + x_speed * t
    if x > 2.2:
        x = 2.2

    # 2. Vertical sinusoidal raster along the car door/side panel (0.30m to 0.80m)
    # Perfectly centers within the UR10e optimal industrial reach envelope
    z = 0.55 + 0.25 * np.sin(2.0 * np.pi * z_freq * t)

    # 3. Y-Axis depth with body curvature compensation
    # Car side panel is at Y = -0.95m. Standoff distance = 250mm (0.25m) -> Base Y = -1.20m
    z_norm = (z - 0.30) / 0.50  # Normalized 0.0 to 1.0
    body_curve = 0.04 * np.sin(np.pi * z_norm)  # Convex door curvature bulge
    y = -1.20 - body_curve

    # 4. Surface-normal orientation matrix
    # Tool Z-axis points towards car (+Y). Dynamic pitch adjusts with vertical slope.
    pitch_deg = -15.0 * (z - 0.55)  # Degrees
    pitch_rad = np.radians(pitch_deg)

    z_axis = np.array([0.0, np.cos(pitch_rad), np.sin(pitch_rad)], dtype=float)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return np.array([x, y, z], dtype=float), rot_matrix


def get_trajectory_duration(x_speed: float = 0.08) -> float:
    """Returns the total duration to sweep the full car length."""
    return (2.2 - (-2.2)) / x_speed
