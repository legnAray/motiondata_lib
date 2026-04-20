from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from motiondata_lib.constants import CANONICAL_JOINT_NAMES
from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import euler_xyz_degrees_to_quat_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef


FORMAT_NAME = "sonic"
DEFAULT_FPS = 120.0
ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
JOINT_COLUMNS = tuple(f"{name}_dof" for name in CANONICAL_JOINT_NAMES)
EXPECTED_COLUMNS = ("Frame", *ROOT_TRANSLATE_COLUMNS, *ROOT_ROTATE_COLUMNS, *JOINT_COLUMNS)


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
    return header and header[0] == "Frame" and all(column in header for column in EXPECTED_COLUMNS)


def load_motion_clip(clip_ref: MotionClipRef, fps_override: float | None = None) -> MotionClip:
    header = _header(clip_ref.path)
    if tuple(header[: len(EXPECTED_COLUMNS)]) != EXPECTED_COLUMNS:
        missing_columns = [column for column in EXPECTED_COLUMNS if column not in header]
        raise ValueError(f"{clip_ref.path} is missing expected Sonic columns: {missing_columns}")

    column_indices = {name: index for index, name in enumerate(header)}
    use_columns = [column_indices[name] for name in EXPECTED_COLUMNS[1:]]
    data = np.loadtxt(clip_ref.path, delimiter=",", skiprows=1, usecols=use_columns, dtype=np.float64)
    data = np.atleast_2d(data)

    return build_motion_clip(
        clip_ref,
        framerate=DEFAULT_FPS,
        joint_names=np.asarray(CANONICAL_JOINT_NAMES),
        joint_pos=np.deg2rad(data[:, 6:]),
        base_pos_w=data[:, 0:3] * 0.01,
        base_quat_w=euler_xyz_degrees_to_quat_wxyz(data[:, 3:6]),
        fps_override=fps_override,
    )
