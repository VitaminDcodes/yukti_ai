"""
main.py

Primary execution entry point for the Automotive Robotic Painting Cell simulation.
Integrates:
- Industrial parametric trajectory generation (controllers/trajectory_generator.py)
- Closed-form Analytical Inverse Kinematics (controllers/ik_solver.py)
- Real-time physical paint deposition engine & 3D visualization (paint_simulation/)
- Real-time physics stepping and collision verification (controllers/collision_checker.py)
- Interactive MuJoCo passive viewer
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

from controllers.ik_solver import UR10eRailController
from controllers.trajectory_generator import get_industrial_curved_waypoint, get_trajectory_duration
from controllers.collision_checker import CollisionChecker
from paint_simulation import SurfacePainter, SprayVisualizer, save_thickness_heatmap


def main():
    print("=" * 72)
    print("  AUTOMOTIVE ROBOTIC PAINTING CELL - MUJOCO SIMULATION")
    print("  Kinematics : Analytical Inverse Kinematics (UR10e + Linear Rail)")
    print("  Deposition : Physical Spray Model + Real-Time Gloss Red Coating")
    import sys
    pattern = 1 if ('--pattern1' in sys.argv or '--pattern=1' in sys.argv) else 2
    pattern_name = "Pattern 1 (Sinusoidal Wave)" if pattern == 1 else "Pattern 2 (Vertical Raster Lines + Sidewise Steps)"
    print(f"  Trajectory : {pattern_name}")
    print("=" * 72)

    # 1. Load MuJoCo model and data
    model = mujoco.MjModel.from_xml_path('models/main_scene.xml')
    data = mujoco.MjData(model)

    # 2. Instantiate controllers and paint simulation engine
    ik_controller = UR10eRailController(model)
    collision_checker = CollisionChecker(model)
    painter = SurfacePainter(model, texture_name="car_paint_tex")
    spray_vis = SprayVisualizer(cone_angle_deg=20.0)

    # 3. Determine starting trajectory waypoint at t=0
    target_pos_0, target_rot_0 = get_industrial_curved_waypoint(0.0, pattern=pattern)

    # Solve initial analytical IK for starting posture (aligned with user-verified configuration)
    initial_q_guess = np.array([2.2, -1.76, -1.0, 0.9, 0.1, -1.76, 0.0])
    q_start = ik_controller.solve_target_pose(target_pos_0, target_rot_0, initial_q_guess)

    # Initialize robot at the starting waypoint
    data.qpos[:] = q_start
    data.ctrl[:] = q_start
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"[INIT] Start Rail Position : {-q_start[0]:.3f} m")
    print(f"[INIT] Start Arm Joints    : {np.round(q_start[1:], 3)}")
    print(f"[INIT] Target TCP Position : {np.round(target_pos_0, 3)} m")
    print(f"[INIT] Paint Grid Size     : {painter.res_x} x {painter.res_z} pixels")
    print(f"[INIT] Active Pattern      : {pattern_name}")
    print("-" * 72)
    print("Launching MuJoCo 3D Viewer... (Press ESC in the viewer to exit)\n")

    traj_duration = get_trajectory_duration(pattern=pattern)
    dt = model.opt.timestep
    sim_time = 0.0
    step_count = 0

    # Smooth 60 FPS display loop with decoupled physics substeps
    render_fps = 60.0
    render_dt = 1.0 / render_fps
    substeps = max(1, int(round(render_dt / dt)))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()

            # Execute physics and coating substeps per visual frame
            for _ in range(substeps):
                # 1. Get current trajectory target waypoint
                target_pos, target_rot = get_industrial_curved_waypoint(sim_time, pattern=pattern)

                # 2. Compute 7-DOF Analytical Inverse Kinematics (Rail + Arm)
                q_target = ik_controller.solve_target_pose(target_pos, target_rot, data.qpos)

                # 3. Apply position control & zero velocity integration drift
                data.ctrl[:] = q_target
                data.qfrc_applied[:] = data.qfrc_bias
                data.qpos[:] = q_target
                data.qvel[:] = 0.0

                # 4. Step physics simulation
                mujoco.mj_step(model, data)

                # 5. Extract active spray nozzle pose
                tcp_world_pos = data.site_xpos[collision_checker.tcp_site_id]
                tcp_rot_mat = data.site_xmat[collision_checker.tcp_site_id].reshape(3, 3)
                spray_dir = tcp_rot_mat[:, 2]  # Tool forward Z-axis points towards car (+Y)

                # 6. Physical Paint Deposition & Dynamic Red Texture Update (CPU buffer)
                painter.update(tcp_world_pos, spray_dir, dt, viewer=None)

                # 7. Collision safety verification
                is_collision, col_msg = collision_checker.check_car_collision(data)
                if is_collision:
                    print(f"[ALERT] {col_msg}")

                sim_time += dt
                step_count += 1

                # Periodic coating telemetry logging (~every 1s of sim time)
                if step_count % int(1.0 / dt) == 0:
                    standoff = collision_checker.get_standoff_distance(data) * 1000.0
                    pos_err = np.linalg.norm(tcp_world_pos - target_pos) * 1000.0
                    metrics = painter.get_metrics()
                    print(
                        f"t={sim_time:5.1f}s | "
                        f"Rail X={-data.qpos[0]:5.2f}m | "
                        f"Standoff={standoff:5.1f}mm | "
                        f"Tracking Err={pos_err:4.2f}mm | "
                        f"Covered={metrics['covered_area_m2']:4.2f}m² | "
                        f"Mean DFT={metrics['mean_thickness_um']:4.1f}µm | "
                        f"Paint={metrics['paint_sprayed_ml']:5.1f}mL"
                    )

            # Re-upload updated texture to GPU once per visual frame (60 FPS)
            viewer.update_texture(painter.tex_id)

            # Clean visualizer: ensure no artificial cylinder in front of nozzle
            spray_vis.draw_spray_cone(viewer, tcp_world_pos, spray_dir, 0.1)

            # Sync viewer display
            viewer.sync()

            # Real-time clock synchronization at 60 FPS
            elapsed = time.time() - frame_start
            sleep_time = render_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # 10. Export final paint deposition heatmap upon simulation completion
    print("\nSimulation ended. Generating coating thickness report...")
    save_thickness_heatmap(painter, filepath="docs/paint_deposition_heatmap.png")


if __name__ == "__main__":
    main()
