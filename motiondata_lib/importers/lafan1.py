from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import quat_xyzw_to_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "lafan1"
DEFAULT_FPS = 30.0


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open(newline="") as handle:
            first_line = handle.readline().strip()
        values = [value.strip() for value in first_line.split(",")]
        if len(values) < 8:
            return False
        [float(value) for value in values]
        return True
    except Exception:
        return False


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
    fps_override: float | None = None,
) -> MotionClip:
    expected_column_count = 7 + len(robot_profile.joint_names)
    data = np.loadtxt(clip_ref.path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != expected_column_count:
        raise ValueError(f"{clip_ref.path} has shape {data.shape}, expected (N, {expected_column_count})")

    return build_motion_clip(
        clip_ref,
        framerate=DEFAULT_FPS,
        joint_names=np.asarray(robot_profile.joint_names),
        joint_pos=data[:, 7:],
        base_pos_w=data[:, 0:3],
        base_quat_w=quat_xyzw_to_wxyz(data[:, 3:7]),
        fps_override=fps_override,
    )
