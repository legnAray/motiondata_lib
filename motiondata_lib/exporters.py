from __future__ import annotations

from pathlib import Path

import numpy as np

from motiondata_lib.types import MotionClip


def export_motion_clip_npz(clip: MotionClip, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        framerate=np.asarray(clip.framerate, dtype=np.float32),
        joint_names=np.asarray(clip.joint_names),
        joint_pos=np.asarray(clip.joint_pos, dtype=np.float32),
        base_pos_w=np.asarray(clip.base_pos_w, dtype=np.float32),
        base_quat_w=np.asarray(clip.base_quat_w, dtype=np.float32),
    )
