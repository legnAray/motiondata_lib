from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


FORMAT_NAME = "retargeted_npz"
STANDARD_NPZ_KEYS = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".npz":
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return STANDARD_NPZ_KEYS.issubset(data.files)
    except Exception:
        return False


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
    fps_override: float | None = None,
) -> MotionClip:
    del robot_profile
    with np.load(clip_ref.path, allow_pickle=False) as data:
        missing = STANDARD_NPZ_KEYS.difference(data.files)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{clip_ref.path} is missing required arrays: {missing_list}")

        return build_motion_clip(
            clip_ref,
            framerate=float(np.asarray(data["framerate"]).item()),
            joint_names=np.asarray(data["joint_names"]),
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float64),
            base_pos_w=np.asarray(data["base_pos_w"], dtype=np.float64),
            base_quat_w=np.asarray(data["base_quat_w"], dtype=np.float64),
            fps_override=fps_override,
        )
