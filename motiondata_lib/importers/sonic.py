from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import euler_xyz_degrees_to_quat_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "sonic"
DEFAULT_FPS = 120.0
ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
REQUIRED_COLUMNS = ("Frame", *ROOT_TRANSLATE_COLUMNS, *ROOT_ROTATE_COLUMNS)


def _header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        return [value.strip() for value in next(reader)]


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        header = _header(path)
    except Exception:
        return False
    return header and header[0] == "Frame" and all(column in header for column in REQUIRED_COLUMNS)


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
) -> MotionClip:
    joint_columns = tuple(f"{name}_dof" for name in robot_profile.joint_names)
    expected_columns = ("Frame", *ROOT_TRANSLATE_COLUMNS, *ROOT_ROTATE_COLUMNS, *joint_columns)
    header = _header(clip_ref.path)
    if tuple(header[: len(expected_columns)]) != expected_columns:
        missing_columns = [column for column in expected_columns if column not in header]
        raise ValueError(f"{clip_ref.path} is missing expected Sonic columns: {missing_columns}")

    column_indices = {name: index for index, name in enumerate(header)}
    use_columns = [column_indices[name] for name in expected_columns[1:]]
    data = np.loadtxt(clip_ref.path, delimiter=",", skiprows=1, usecols=use_columns, dtype=np.float64)
    data = np.atleast_2d(data)

    return build_motion_clip(
        clip_ref,
        framerate=DEFAULT_FPS,
        joint_names=np.asarray(robot_profile.joint_names),
        joint_pos=np.deg2rad(data[:, 6:]),
        base_pos_w=data[:, 0:3] * 0.01,
        base_quat_w=euler_xyz_degrees_to_quat_wxyz(data[:, 3:6]),
    )
