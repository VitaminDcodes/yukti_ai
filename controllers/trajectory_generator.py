"""
controllers/trajectory_generator.py

Parametric industrial trajectory generation engine for automotive robotic painting cell.
Supports:
- Pattern 1: Continuous sinusoidal raster wave (smooth continuous passes).
- Pattern 2: Boustrophedon vertical line raster with sidewise steps (as shown in industrial painting spec).
Both patterns include car profile height adaptation (taller in middle, lower at ends),
body curvature compensation, and surface-normal pitch compensation.
"""

import numpy as np


def get_car_profile_bounds(x: float):
    """
    Computes profile-adaptive vertical bounds [z_bottom, z_top] matching vehicle profile:
    - Downward reach: z_bottom = 0.25m (coats rocker panel/skirt all the way down).
    - Rail ends (front hood x ~ -2.2m, rear trunk x ~ +2.2m): lower car profile (z_top ~ 0.78m - 0.84m).
    - Middle cabin/doors (x in [-0.9m, +0.9m]): taller car profile (z_top ~ 1.05m).
    """
    z_bottom = 0.25
    cabin_peak = 0.26 * np.exp(-0.5 * (x / 0.95) ** 2)
    base_top = 0.78 + 0.06 * ((x + 2.2) / 4.4)
    z_top = base_top + cabin_peak
    return z_bottom, z_top


def get_pattern1_waypoint(t: float, x_speed: float = 0.35, z_freq: float = 0.45):
    """
    Pattern 1: Continuous sinusoidal wave trajectory with profile-adaptive height.
    Sweeps along car length while oscillating vertically across the full car profile.
    
    Parameters:
    - t: Simulation time (seconds)
    - x_speed: Travel speed along car length (m/s)
    - z_freq: Vertical oscillation frequency (Hz)
    
    Returns:
    - target_pos: np.ndarray [x, y, z] target TCP position in world coordinates
    - rot_matrix: np.ndarray (3, 3) orientation matrix keeping spray nozzle normal to panel
    """
    # 1. Continuous bidirectional longitudinal sweep along car
    x_min, x_max = -2.2, 2.2
    span = x_max - x_min
    phase = (t * x_speed) % (2.0 * span)
    if phase <= span:
        x = x_min + phase
    else:
        x = x_max - (phase - span)

    # 2. Vertical sinusoidal raster spanning full car height (lower at ends, taller at center)
    z_b, z_t = get_car_profile_bounds(x)
    z_mid = 0.5 * (z_b + z_t)
    z_amp = 0.5 * (z_t - z_b)
    z = z_mid + z_amp * np.sin(2.0 * np.pi * z_freq * t)

    # 3. Y-Axis depth with body curvature and taper compensation
    z_norm = (z - 0.25) / 0.80
    door_curve = 0.02 * np.sin(np.pi * z_norm)
    fender_taper = 0.06 * (x / 2.2) ** 2
    y = -1.05 + fender_taper - door_curve

    # 4. Surface-normal orientation matrix
    pitch_deg = -15.0 * (z - 0.65)
    pitch_rad = np.radians(pitch_deg)

    z_axis = np.array([0.0, np.cos(pitch_rad), np.sin(pitch_rad)], dtype=float)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return np.array([x, y, z], dtype=float), rot_matrix


def get_pattern2_waypoint(
    t: float,
    v_vert: float = 0.45,
    v_step: float = 0.35,
    step_dx: float = 0.10,
    x_start: float = -2.2,
    x_end: float = 2.2
):
    """
    Pattern 2: Boustrophedon vertical line raster with sidewise steps and profile-adaptive height.
    Paints vertically in a straight line from car top to bottom (or bottom to top),
    steps horizontally sideways, and repeats across the vehicle.
    
    Parameters:
    - t: Simulation time (seconds)
    - v_vert: Linear vertical stroke velocity (m/s)
    - v_step: Horizontal step velocity (m/s)
    - step_dx: Sidewise step distance between passes (m)
    - x_start, x_end: Longitudinal bounds along the car (m)
    
    Returns:
    - target_pos: np.ndarray [x, y, z] target TCP position in world coordinates
    - rot_matrix: np.ndarray (3, 3) orientation matrix keeping spray nozzle normal to panel
    """
    total_steps = int(np.round((x_end - x_start) / step_dx))
    # Nominal stroke timing
    t_vert = 0.65 / v_vert
    t_step = step_dx / v_step
    t_segment = t_vert + t_step
    t_one_way = total_steps * t_segment

    # Continuous bidirectional loop across car length
    t_mod = t % (2.0 * t_one_way)
    t_dir = t_mod if t_mod <= t_one_way else (2.0 * t_one_way - t_mod)

    idx = min(total_steps - 1, int(t_dir // t_segment))
    rem = t_dir - idx * t_segment
    moving_up = (idx % 2 == 0)
    x_base = x_start + idx * step_dx

    if rem < t_vert:
        # Vertical straight line pass
        s = rem / t_vert
        x = x_base
        z_b, z_t = get_car_profile_bounds(x)
        z = z_b + s * (z_t - z_b) if moving_up else z_t - s * (z_t - z_b)
    else:
        # Sidewise step with smooth S-curve transition
        s = (rem - t_vert) / t_step
        s_smooth = s * s * (3.0 - 2.0 * s)
        x = x_base + s_smooth * step_dx
        z_b, z_t = get_car_profile_bounds(x)
        z = z_t if moving_up else z_b

    # Y-Axis depth with body curvature and taper compensation
    z_norm = (z - 0.25) / 0.80
    door_curve = 0.02 * np.sin(np.pi * z_norm)
    fender_taper = 0.06 * (x / 2.2) ** 2
    y = -1.05 + fender_taper - door_curve

    # Surface-normal orientation matrix
    pitch_deg = -15.0 * (z - 0.65)
    pitch_rad = np.radians(pitch_deg)

    z_axis = np.array([0.0, np.cos(pitch_rad), np.sin(pitch_rad)], dtype=float)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return np.array([x, y, z], dtype=float), rot_matrix


def get_industrial_curved_waypoint(t: float, pattern: int = 2, **kwargs):
    """
    Primary interface for trajectory waypoints.
    Defaults to Pattern 2 (vertical raster with sidewise steps).
    Set pattern=1 for continuous sinusoidal wave.
    """
    if pattern == 1:
        return get_pattern1_waypoint(t, **kwargs)
    else:
        return get_pattern2_waypoint(t, **kwargs)


def get_trajectory_duration(pattern: int = 2, **kwargs) -> float:
    """Returns single-pass duration for the selected pattern."""
    if pattern == 1:
        x_speed = kwargs.get('x_speed', 0.35)
        return 4.4 / x_speed
    else:
        v_vert = kwargs.get('v_vert', 0.45)
        v_step = kwargs.get('v_step', 0.35)
        step_dx = kwargs.get('step_dx', 0.10)
        t_segment = (0.65 / v_vert) + (step_dx / v_step)
        return int(np.round(4.4 / step_dx)) * t_segment
