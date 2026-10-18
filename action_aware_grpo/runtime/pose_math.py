"""WorldVLN action contract: previous-body SE(3), centimeters and degrees.

Pose/action order is [x, y, z, roll, yaw, pitch]. Euler convention matches
training labels: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). A delta is composed on
the right: T_next = T_previous @ T_delta (translation uses previous R).
"""
from __future__ import annotations

import numpy as np

ACTION_CONTRACT = {
    "version": "worldvln.previous_body_se3.v1",
    "frame": "previous_body",
    "composition": "T_next=T_previous@T_delta",
    "body_apply_order": "trans_first",
    "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
    "action_order": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
    "translation_unit": "cm",
    "angle_unit": "deg",
    "euler_convention": "Rz(yaw)@Ry(pitch)@Rx(roll)",
    "world_poses_source": "simulator_observed",
}


def pose_to_matrix(pose_cmdeg):
    p = np.asarray(pose_cmdeg, dtype=np.float64)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError("pose/action must contain exactly six finite numbers")
    roll, yaw, pitch = np.deg2rad(p[3:6])
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    out[:3, 3] = p[:3]
    return out


def matrix_to_pose(matrix):
    t = np.asarray(matrix, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("matrix must be finite with shape (4,4)")
    r = t[:3, :3]
    pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
    if np.hypot(r[0, 0], r[1, 0]) > 1e-10:
        yaw = np.arctan2(r[1, 0], r[0, 0])
        roll = np.arctan2(r[2, 1], r[2, 2])
    else:
        yaw = np.arctan2(-r[0, 1], r[1, 1])
        roll = 0.0
    return t[:3, 3].tolist() + np.rad2deg([roll, yaw, pitch]).tolist()


def compose_body_delta(pose, delta):
    return matrix_to_pose(pose_to_matrix(pose) @ pose_to_matrix(delta))


def integrate_body_actions(actions, initial_pose=None):
    pose = [0.0] * 6 if initial_pose is None else list(initial_pose)
    pose_to_matrix(pose)
    poses = [pose]
    for action in actions:
        pose = compose_body_delta(pose, action)
        poses.append(pose)
    return poses


def world_poses_to_initial(poses):
    poses = list(poses)
    if not poses:
        return []
    start_inv = np.linalg.inv(pose_to_matrix(poses[0]))
    return [matrix_to_pose(start_inv @ pose_to_matrix(pose)) for pose in poses]
