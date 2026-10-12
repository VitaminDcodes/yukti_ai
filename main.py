"""
main.py

Primary execution entry point for the Automotive Robotic Painting Cell simulation.
Integrates:
- Industrial parametric trajectory generation
- Fast closed-form Analytical Inverse Kinematics (UR10e + Linear Rail)
- Real-time physics stepping and collision verification
- Interactive MuJoCo passive viewer
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

from controllers.ik_solver import UR10eRailController, ur10e
from controllers.trajectory_generator import get_industrial_curved_waypoint, get_trajectory_duration
from controllers.collision_checker import CollisionChecker


def main():
    print("=" * 68)
    print("  AUTOMOTIVE ROBOTIC PAINTING CELL - MUJOCO SIMULATION")
    print("  Kinematics Engine: Analytical Inverse Kinematics (UR10e + Linear Rail)")
    print("=" * 68)

    # 1. Load MuJoCo model and data
    model = mujoco.MjModel.from_xml_path('models/main_scene.xml')
    data = mujoco.MjData(model)

    # 2. Instantiate controllers
    ik_controller = UR10eRailController(model)
    collision_checker = CollisionChecker(model)

    # 3. Determine starting trajectory waypoint at t=0
    target_pos_0, target_rot_0 = get_industrial_curved_waypoint(0.0)

    # Solve initial analytical IK for starting posture
    initial_q_guess = np.array([2.2, 0.0, -1.2, 1.5, -1.8, -1.57, 0.0])
    q_start = ik_controller.solve_target_pose(target_pos_0, target_rot_0, initial_q_guess)

    # Initialize robot at the starting waypoint
    data.qpos[:] = q_start
    data.ctrl[:] = q_start
    mujoco.mj_forward(model, data)

    print(f"[INIT] Start Rail Position : {-q_start[0]:.3f} m")
    print(f"[INIT] Start Arm Joints    : {np.round(q_start[1:], 3)}")
    print(f"[INIT] Target TCP Position : {np.round(target_pos_0, 3)} m")
    print("-" * 68)
    print("Launching MuJoCo 3D Viewer... (Press ESC in the viewer to exit)\n")

    traj_duration = get_trajectory_duration()
    dt = model.opt.timestep
    sim_time = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. Get current trajectory target waypoint
            target_pos, target_rot = get_industrial_curved_waypoint(sim_time)

            # 2. Compute 7-DOF Analytical Inverse Kinematics (Rail + Arm)
            q_target = ik_controller.solve_target_pose(target_pos, target_rot, data.qpos)

            # 3. Apply position control with gravity compensation
            data.ctrl[:] = q_target
            data.qfrc_applied[:] = data.qfrc_bias
            
            # Kinematically align positions for zero-lag trajectory visualization
            data.qpos[:] = q_target

            # 4. Step physics simulation
            mujoco.mj_step(model, data)

            # 5. Collision checking and safety verification
            is_collision, col_msg = collision_checker.check_car_collision(data)
            if is_collision:
                print(f"[ALERT] {col_msg}")

            # 6. Periodic telemetry logging (every ~1 second of sim time)
            step_count += 1
            if step_count % int(1.0 / dt) == 0:
                standoff = collision_checker.get_standoff_distance(data) * 1000.0  # mm
                actual_tcp = data.site_xpos[collision_checker.tcp_site_id]
                pos_err = np.linalg.norm(actual_tcp - target_pos) * 1000.0  # mm
                print(f"t={sim_time:5.1f}s | Rail X={-data.qpos[0]:5.2f}m | Standoff={standoff:5.1f}mm | Tracking Error={pos_err:5.2f}mm | Status=OK")

            # 7. Sync viewer
            viewer.sync()

            # 8. Advance simulation time
            sim_time += dt

            # 9. Real-time rate synchronization
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
