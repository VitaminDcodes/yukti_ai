"""
controllers/collision_checker.py

Real-time collision detection and standoff distance monitoring for the
automotive painting cell in MuJoCo.
"""

import mujoco
import numpy as np


class CollisionChecker:
    """
    Monitors data.ncon to detect unintended collisions between the robot arm/tool
    and the car workpiece, and computes tool-to-surface standoff distance.
    """

    def __init__(self, model):
        self.model = model
        self.car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        # Collect all robot body IDs
        self.robot_body_ids = set()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if any(k in name.lower() for k in ["base", "shoulder", "upper_arm", "forearm", "wrist", "painting_tool"]):
                self.robot_body_ids.add(i)

    def check_car_collision(self, data) -> tuple[bool, str]:
        """
        Returns (is_collision, message).
        Detects if any part of the robot arm or spray tool contacts the car body.
        """
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            # Check if one body is the car and the other is part of the robot
            car_involved = (b1 == self.car_body_id or b2 == self.car_body_id)
            if car_involved:
                other_body = b2 if b1 == self.car_body_id else b1
                if other_body in self.robot_body_ids:
                    b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other_body)
                    return True, f"COLLISION: Robot body '{b_name}' collided with Car!"

        return False, "Clear"

    def get_standoff_distance(self, data) -> float:
        """
        Computes the distance from the TCP nozzle tip to the car side panel plane (Y = -0.95m).
        Returns distance in meters.
        """
        tcp_world_pos = data.site_xpos[self.tcp_site_id]
        # Car side panel is at Y = -0.95m in world coordinates
        car_panel_y = -0.95
        standoff = abs(tcp_world_pos[1] - car_panel_y)
        return float(standoff)
