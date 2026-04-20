from __future__ import annotations

from pathlib import Path

import numpy as np

from motiondata_lib.constants import CANONICAL_JOINT_NAMES
from motiondata_lib.importers.common import build_motion_clip
from motiondata_lib.transforms import quat_xyzw_to_wxyz
from motiondata_lib.types import MotionClip, MotionClipRef


FORMAT_NAME = "lafan1"
DEFAULT_FPS = 30.0
EXPECTED_COLUMN_COUNT = 36


def can_load(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open(newline="") as handle:
            first_line = handle.readline().strip()
        values = [value.strip() for value in first_line.split(",")]
        if len(values) != EXPECTED_COLUMN_COUNT:
            return False
        [float(value) for value in values]
        return True
    except Exception:
        return False


def load_motion_clip(clip_ref: MotionClipRef, fps_override: float | None = None) -> MotionClip:
    data = np.loadtxt(clip_ref.path, delimiter=",", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(f"{clip_ref.path} has shape {data.shape}, expected (N, {EXPECTED_COLUMN_COUNT})")

    return build_motion_clip(
        clip_ref,
        framerate=DEFAULT_FPS,
        joint_names=np.asarray(CANONICAL_JOINT_NAMES),
        joint_pos=data[:, 7:],
        base_pos_w=data[:, 0:3],
        base_quat_w=quat_xyzw_to_wxyz(data[:, 3:7]),
        fps_override=fps_override,
    )
