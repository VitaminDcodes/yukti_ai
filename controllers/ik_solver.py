"""
controllers/ik_solver.py

High-performance closed-form Analytical Inverse Kinematics (IK) for the UR10e
manipulator and 1D linear rail system in MuJoCo.

Provides:
- ur10e.forward_kinematics(*joints) -> 4x4 matrix
- ur10e.inverse_kinematics(T_target) -> list of up to 8 solutions
- ur10e.inverse_kinematics_closest(T_target, *q_current) -> closest 6-joint solution
- UR10eRailController: 7-DOF controller coordinating linear rail and UR10e arm
"""

import math
import numpy as np

# Calibrated DH parameters for UR10e in MuJoCo
DH_d1 = 0.181
DH_a2 = -0.613
DH_a3 = -0.571
DH_d4 = 0.174
DH_d5 = 0.120
DH_d6 = 0.100  # Distance from wrist_3 to tool flange (attachment_site)

ZERO_THRESH = 1e-8
PI = math.pi

# Fixed orientation offset between DH Frame 6 and MuJoCo attachment_site
R_DH_TO_ATT = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0]
], dtype=float)

# Transform from attachment_site to TCP site in MuJoCo
# attachment_site -> painting_tool -> tcp site (aligned along tool Z towards car)
T_ATT_TO_TCP = np.array([
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0, -0.23],
    [ 0.0,  0.0,  0.0,  1.0]
], dtype=float)

T_TCP_TO_ATT = np.linalg.inv(T_ATT_TO_TCP)


def _sign(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def _norm_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (a + PI) % (2.0 * PI) - PI


class _UR10eKinematics:
    """
    Closed-form analytical kinematics solver for UR10e.
    Computes all 8 exact geometric IK solutions with machine precision (< 1e-12).
    """

    @staticmethod
    def forward_kinematics(*joint_angles) -> np.ndarray:
        """
        Computes forward kinematics for UR10e given 6 joint angles (radians).
        Returns a 4x4 homogeneous transformation matrix of attachment_site relative to base.
        """
        if len(joint_angles) == 1 and hasattr(joint_angles[0], '__len__'):
            q = list(joint_angles[0])
        else:
            q = list(joint_angles)

        s1, c1 = math.sin(q[0]), math.cos(q[0])
        q234 = q[1] + q[2] + q[3]
        s2, c2 = math.sin(q[1]), math.cos(q[1])
        s3, c3 = math.sin(q[2]), math.cos(q[2])
        s5, c5 = math.sin(q[4]), math.cos(q[4])
        s6, c6 = math.sin(q[5]), math.cos(q[5])
        s234, c234 = math.sin(q234), math.cos(q234)

        T_dh = np.zeros((4, 4), dtype=float)
        T_dh[0, 0] = ((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0
        T_dh[0, 1] = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0)
        T_dh[0, 2] = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0))
        T_dh[0, 3] = ((DH_d5*(s1*c234-c1*s234))/2.0 - (DH_d5*(s1*c234+c1*s234))/2.0 - DH_d4*s1 + (DH_d6*(c1*c234-s1*s234)*s5)/2.0 + (DH_d6*(c1*c234+s1*s234)*s5)/2.0 - DH_a2*c1*c2 - DH_d6*c5*s1 - DH_a3*c1*c2*c3 + DH_a3*c1*s2*s3)

        T_dh[1, 0] = c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0
        T_dh[1, 1] = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0))
        T_dh[1, 2] = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0))
        T_dh[1, 3] = ((DH_d5*(c1*c234-s1*s234))/2.0 - (DH_d5*(c1*c234+s1*s234))/2.0 + DH_d4*c1 + (DH_d6*(s1*c234+c1*s234)*s5)/2.0 + (DH_d6*(s1*c234-c1*s234)*s5)/2.0 + DH_d6*c1*c5 - DH_a2*c2*s1 - DH_a3*c2*c3*s1 + DH_a3*s1*s2*s3)

        T_dh[2, 0] = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0)
        T_dh[2, 1] = ((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6)
        T_dh[2, 2] = (s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0)
        T_dh[2, 3] = (DH_d1 + (DH_d6*(c234*c5-s234*s5))/2.0 + DH_a3*(s2*c3+c2*s3) + DH_a2*s2 - (DH_d6*(c234*c5+s234*s5))/2.0 - DH_d5*c234)
        T_dh[3, 3] = 1.0

        # Map DH orientation to MuJoCo attachment_site orientation
        T_att = np.copy(T_dh)
        T_att[:3, :3] = T_dh[:3, :3] @ R_DH_TO_ATT
        return T_att

    @staticmethod
    def inverse_kinematics(T_target: np.ndarray, q6_des: float = 0.0) -> list[np.ndarray]:
        """
        Computes all closed-form analytical IK solutions for UR10e.
        T_target: 4x4 matrix of attachment_site pose relative to robot base.
        Returns a list of 6-element numpy arrays.
        """
        # Convert MuJoCo attachment_site pose to DH convention
        T_dh = np.copy(T_target)
        T_dh[:3, :3] = T_target[:3, :3] @ R_DH_TO_ATT.T

        T00, T01, T02, T03 = T_dh[0, 1], T_dh[0, 2], -T_dh[0, 0], -T_dh[0, 3]
        T10, T11, T12, T13 = T_dh[1, 1], T_dh[1, 2], -T_dh[1, 0], -T_dh[1, 3]
        T20, T21, T22, T23 = -T_dh[2, 1], -T_dh[2, 2], T_dh[2, 0], T_dh[2, 3]

        A = DH_d6 * T12 - T13
        B = DH_d6 * T02 - T03
        R = A * A + B * B
        if DH_d4 * DH_d4 > R:
            return []

        arccos = math.acos(max(-1.0, min(1.0, DH_d4 / math.sqrt(R))))
        arctan = math.atan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        q1 = [pos if pos >= 0.0 else 2.0 * PI + pos, neg if neg >= 0.0 else 2.0 * PI + neg]

        q_sols = []
        for i in range(2):
            numer = T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - DH_d4
            val = max(-1.0, min(1.0, numer / DH_d6))
            arccos5 = math.acos(val)
            q5_opts = [arccos5, 2.0 * PI - arccos5]

            for j in range(2):
                q5_val = q5_opts[j]
                c1, s1 = math.cos(q1[i]), math.sin(q1[i])
                c5, s5 = math.cos(q5_val), math.sin(q5_val)

                if abs(s5) < ZERO_THRESH:
                    q6 = q6_des
                else:
                    q6 = math.atan2(_sign(s5) * -(T01 * s1 - T11 * c1), _sign(s5) * (T00 * s1 - T10 * c1))
                    if q6 < 0.0:
                        q6 += 2.0 * PI

                c6, s6 = math.cos(q6), math.sin(q6)
                x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
                x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
                p13x = DH_d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - DH_d6 * (T02 * c1 + T12 * s1) + T03 * c1 + T13 * s1
                p13y = T23 - DH_d1 - DH_d6 * T22 + DH_d5 * (T21 * c6 + T20 * s6)

                c3 = (p13x * p13x + p13y * p13y - DH_a2 * DH_a2 - DH_a3 * DH_a3) / (2.0 * DH_a2 * DH_a3)
                if abs(c3) > 1.0:
                    continue
                c3 = max(-1.0, min(1.0, c3))
                arccos3 = math.acos(c3)
                q3_opts = [arccos3, 2.0 * PI - arccos3]

                denom = DH_a2 * DH_a2 + DH_a3 * DH_a3 + 2.0 * DH_a2 * DH_a3 * c3
                s3 = math.sin(arccos3)
                A_q2 = DH_a2 + DH_a3 * c3
                B_q2 = DH_a3 * s3
                q2_opts = [
                    math.atan2((A_q2 * p13y - B_q2 * p13x) / denom, (A_q2 * p13x + B_q2 * p13y) / denom),
                    math.atan2((A_q2 * p13y + B_q2 * p13x) / denom, (A_q2 * p13x - B_q2 * p13y) / denom)
                ]

                for k in range(2):
                    q2_val = q2_opts[k]
                    q3_val = q3_opts[k]
                    c23 = math.cos(q2_val + q3_val)
                    s23 = math.sin(q2_val + q3_val)
                    q4_val = math.atan2(c23 * x04y - s23 * x04x, x04x * c23 + x04y * s23)

                    sol = np.array([
                        _norm_angle(q1[i]),
                        _norm_angle(q2_val),
                        _norm_angle(q3_val),
                        _norm_angle(q4_val),
                        _norm_angle(q5_val),
                        _norm_angle(q6)
                    ], dtype=float)
                    q_sols.append(sol)

        return q_sols

    @classmethod
    def inverse_kinematics_closest(cls, T_target: np.ndarray, *q_current) -> np.ndarray | None:
        """
        Finds the analytical solution that minimizes angular deviation from q_current.
        Guarantees smooth trajectory following without joint flips.
        """
        if len(q_current) == 1 and hasattr(q_current[0], '__len__'):
            q_cur = np.array(q_current[0], dtype=float)
        else:
            q_cur = np.array(q_current, dtype=float)

        sols = cls.inverse_kinematics(T_target)
        if not sols:
            return None

        best_sol = None
        min_cost = float('inf')

        # Weights penalizing shoulder and elbow jumps more than wrist spin
        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])

        for sol in sols:
            diff = np.array([_norm_angle(s - c) for s, c in zip(sol, q_cur)])
            cost = np.sum(weights * (diff ** 2))
            if cost < min_cost:
                min_cost = cost
                # Unwrap solution continuous to q_cur
                best_sol = np.array([c + d for c, d in zip(q_cur, diff)])

        return best_sol


# Export as ur10e singleton module for exact API compatibility
ur10e = _UR10eKinematics


class UR10eRailController:
    """
    Coordinated 7-DOF Inverse Kinematics Controller:
    - 1-DOF Linear rail track along the car (X-axis)
    - 6-DOF UR10e articulated arm
    """

    def __init__(self, model, rail_limits=(-2.7, 2.7)):
        self.model = model
        self.rail_min, self.rail_max = rail_limits

    def solve_target_pose(self, target_world_pos: np.ndarray, target_world_rot: np.ndarray, q_current: np.ndarray):
        """
        Solves 7-DOF IK (rail + arm) for a desired TCP pose in world coordinates.
        target_world_pos: [x, y, z] target in world frame
        target_world_rot: 3x3 rotation matrix of TCP in world frame
        q_current: [q_rail, q1, q2, q3, q4, q5, q6] current configuration
        
        Returns:
            q_target: 7-element array [q_rail, q1, q2, q3, q4, q5, q6]
        """
        target_world_pos = np.asarray(target_world_pos, dtype=float)
        target_world_rot = np.asarray(target_world_rot, dtype=float)

        # 1. Coordinate linear rail position with target X
        # Base body has quat=[0, 0, 0, -1] which flips local X relative to world X
        # Rail slider position q0 = -world_x
        desired_rail_pos = float(np.clip(-target_world_pos[0], self.rail_min, self.rail_max))

        # Base world frame:
        # base_pos = [-desired_rail_pos, -2.1, 0.0]
        # Base orientation: 180 deg around Z
        R_base_world = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ])
        p_base_world = np.array([-desired_rail_pos, -2.1, 0.0])

        # 2. Convert desired TCP pose from world to base frame
        T_tcp_world = np.eye(4)
        T_tcp_world[:3, :3] = target_world_rot
        T_tcp_world[:3, 3] = target_world_pos

        T_base_world = np.eye(4)
        T_base_world[:3, :3] = R_base_world
        T_base_world[:3, 3] = p_base_world

        T_tcp_base = np.linalg.inv(T_base_world) @ T_tcp_world

        # 3. Compute required attachment_site pose from TCP pose
        T_att_base = T_tcp_base @ T_TCP_TO_ATT

        # 4. Analytical Inverse Kinematics for the 6-DOF arm
        arm_current = q_current[1:7]
        arm_sol = ur10e.inverse_kinematics_closest(T_att_base, arm_current)

        if arm_sol is None:
            # Fallback: keep current arm posture if out of workspace
            return q_current.copy()

        q_target = np.zeros(7, dtype=float)
        q_target[0] = desired_rail_pos
        q_target[1:7] = arm_sol
        return q_target
