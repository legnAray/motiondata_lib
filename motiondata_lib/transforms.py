from __future__ import annotations

import numpy as np


def normalize_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    return quat_wxyz / np.clip(norms, 1e-8, None)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return normalize_quaternions(quat_xyzw[:, [3, 0, 1, 2]])


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def euler_xyz_degrees_to_quat_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    half_angles = 0.5 * euler_rad
    sin_half = np.sin(half_angles)
    cos_half = np.cos(half_angles)

    count = len(euler_deg)
    qx = np.stack((cos_half[:, 0], sin_half[:, 0], np.zeros(count), np.zeros(count)), axis=-1)
    qy = np.stack((cos_half[:, 1], np.zeros(count), sin_half[:, 1], np.zeros(count)), axis=-1)
    qz = np.stack((cos_half[:, 2], np.zeros(count), np.zeros(count), sin_half[:, 2]), axis=-1)

    quat_wxyz = quat_multiply(qz, quat_multiply(qy, qx))
    return normalize_quaternions(quat_wxyz)
