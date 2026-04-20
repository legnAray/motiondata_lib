from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionClipRef:
    path: Path
    display_name: str
    format_name: str


@dataclass(frozen=True)
class MotionClip:
    path: Path
    display_name: str
    format_name: str
    framerate: float
    joint_names: np.ndarray
    joint_pos: np.ndarray
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])
