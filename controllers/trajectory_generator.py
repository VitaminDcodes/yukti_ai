"""
controllers/trajectory_generator.py

Parametric industrial surface-normal trajectory generator for automotive painting cell.
Generates smooth 3D tool center point (TCP) waypoints and orientation matrices
maintaining constant standoff distance (250mm) and dynamic normal pitch compensation.
"""

import numpy as np


def get_industrial_curved_waypoint(t: float, x_speed: float = 0.35, z_freq: float = 0.45):
    """
    Generates target 3D TCP position and orientation matching local car surface curvature.
    
    Parameters:
    - t: Current simulation time (seconds)
    - x_speed: Travel speed along the car length (m/s)
    - z_freq: Vertical raster frequency (Hz)
    
    Returns:
    - target_pos: np.ndarray [x, y, z] target TCP position in world coordinates
    - rot_matrix: np.ndarray (3, 3) orientation matrix keeping spray nozzle normal to panel
    """
    # 1. Continuous bidirectional longitudinal sweep along car (car spans -2.4m to +2.4m)
    x_min = -2.2
    x_max = 2.2
    span = x_max - x_min
    phase = (t * x_speed) % (2.0 * span)
    if phase <= span:
        x = x_min + phase
    else:
        x = x_max - (phase - span)

    # 2. Vertical sinusoidal raster along the car door/side panel (0.40m to 0.90m)
    z = 0.65 + 0.25 * np.sin(2.0 * np.pi * z_freq * t)

    # 3. Y-Axis depth with body curvature compensation
    # Car surface is at Y = -0.94m (doors) tapering to Y = -0.88m (fenders).
    # Standoff distance = 100mm -> Base Y = -1.05m with fender taper adjustment
    z_norm = (z - 0.40) / 0.50
    door_curve = 0.02 * np.sin(np.pi * z_norm)
    fender_taper = 0.06 * (x / 2.2) ** 2
    y = -1.05 + fender_taper - door_curve

    # 4. Surface-normal orientation matrix
    # Tool Z-axis points towards car (+Y). Dynamic pitch adjusts with vertical slope.
    pitch_deg = -12.0 * (z - 0.65)
    pitch_rad = np.radians(pitch_deg)

    z_axis = np.array([0.0, np.cos(pitch_rad), np.sin(pitch_rad)], dtype=float)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return np.array([x, y, z], dtype=float), rot_matrix


def get_trajectory_duration(x_speed: float = 0.35) -> float:
    """Returns the single-pass duration along the car length."""
    return (2.2 - (-2.2)) / x_speed
