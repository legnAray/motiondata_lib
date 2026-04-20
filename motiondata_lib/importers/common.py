from __future__ import annotations

from pathlib import Path

import numpy as np

from motiondata_lib.transforms import normalize_quaternions
from motiondata_lib.types import MotionClip, MotionClipRef


SUPPORTED_MOTION_SUFFIXES = {".npz", ".csv", ".npy"}


def motion_files_under_dir(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_MOTION_SUFFIXES
    )


def validate_motion_clip_arrays(
    path: Path,
    joint_names: np.ndarray,
    joint_pos: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
) -> None:
    frame_count = joint_pos.shape[0]
    if frame_count == 0:
        raise ValueError(f"{path} does not contain any frames")
    if joint_pos.ndim != 2:
        raise ValueError(f"{path} has joint_pos shape {joint_pos.shape}, expected a 2D array")
    if base_pos_w.shape != (frame_count, 3):
        raise ValueError(f"{path} has base_pos_w shape {base_pos_w.shape}, expected ({frame_count}, 3)")
    if base_quat_w.shape != (frame_count, 4):
        raise ValueError(f"{path} has base_quat_w shape {base_quat_w.shape}, expected ({frame_count}, 4)")
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(f"{path} has {joint_pos.shape[1]} joint columns but {len(joint_names)} joint names")


def build_motion_clip(
    clip_ref: MotionClipRef,
    *,
    framerate: float,
    joint_names: np.ndarray,
    joint_pos: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
    fps_override: float | None = None,
) -> MotionClip:
    if fps_override is not None:
        framerate = fps_override

    joint_names = np.asarray(joint_names)
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    base_quat_w = normalize_quaternions(np.asarray(base_quat_w, dtype=np.float64))

    validate_motion_clip_arrays(clip_ref.path, joint_names, joint_pos, base_pos_w, base_quat_w)

    return MotionClip(
        path=clip_ref.path,
        display_name=clip_ref.display_name,
        format_name=clip_ref.format_name,
        framerate=framerate if framerate > 0 else 30.0,
        joint_names=joint_names,
        joint_pos=joint_pos,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
    )
