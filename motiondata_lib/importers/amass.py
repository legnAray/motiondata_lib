from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import quat_xyzw_to_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "amass"
BASE_Z_OFFSET = 0.75


def _extract_fps(path: Path) -> float:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Could not infer AMASS frame rate from filename: {path.name}")
    return float(matches[-1])


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".npy":
        return False
    try:
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        return data.ndim == 2 and data.shape[1] >= 8
    except Exception:
        return False


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
    fps_override: float | None = None,
) -> MotionClip:
    expected_column_count = 7 + len(robot_profile.joint_names)
    data = np.load(clip_ref.path, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != expected_column_count:
        raise ValueError(f"{clip_ref.path} has shape {data.shape}, expected (N, {expected_column_count})")

    base_pos_w = np.array(data[:, 0:3], copy=True)
    base_pos_w[:, 2] += BASE_Z_OFFSET

    return build_motion_clip(
        clip_ref,
        framerate=_extract_fps(clip_ref.path),
        joint_names=np.asarray(robot_profile.joint_names),
        joint_pos=data[:, 7:],
        base_pos_w=base_pos_w,
        base_quat_w=quat_xyzw_to_wxyz(data[:, 3:7]),
        fps_override=fps_override,
    )
