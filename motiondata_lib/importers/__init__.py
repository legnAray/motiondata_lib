from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from motiondata_lib.importers import amass, lafan1, retargeted_npz, sonic
from motiondata_lib.importers.common import motion_files_under_dir
from motiondata_lib.types import MotionClip, MotionClipRef

if TYPE_CHECKING:
    from motiondata_lib.robot_profiles import RobotProfile


IMPORTERS = (retargeted_npz, sonic, lafan1, amass)
IMPORTER_BY_NAME = {importer.FORMAT_NAME: importer for importer in IMPORTERS}
SUPPORTED_DATASET_FORMATS = tuple(importer.FORMAT_NAME for importer in IMPORTERS)


def detect_motion_file_format(path: Path) -> str:
    for importer in IMPORTERS:
        if importer.can_load(path):
            return importer.FORMAT_NAME
    raise ValueError(f"Unsupported motion file: {path}")


def detect_dataset_format(dataset_dir: Path) -> str:
    motion_files = motion_files_under_dir(dataset_dir)
    if not motion_files:
        raise ValueError(f"No supported motion files were found under {dataset_dir}")

    formats = {detect_motion_file_format(path) for path in motion_files}
    if len(formats) != 1:
        raise ValueError(f"Expected a single dataset format under {dataset_dir}, found: {sorted(formats)}")
    return formats.pop()


def discover_motion_clips(dataset_dir: Path, format_hint: str = "auto") -> list[MotionClipRef]:
    dataset_dir = dataset_dir.resolve()
    if format_hint == "auto":
        dataset_format = detect_dataset_format(dataset_dir)
    else:
        if format_hint not in IMPORTER_BY_NAME:
            raise ValueError(f"Unsupported dataset format '{format_hint}'")
        dataset_format = format_hint

    importer = IMPORTER_BY_NAME[dataset_format]
    clips = []
    for path in motion_files_under_dir(dataset_dir):
        if not importer.can_load(path):
            continue
        relative = path.relative_to(dataset_dir).with_suffix("")
        clips.append(MotionClipRef(path=path, display_name="/".join(relative.parts), format_name=dataset_format))

    if not clips:
        raise ValueError(f"No {dataset_format} motion files were found under {dataset_dir}")
    return clips


def load_motion_clip(
    clip_ref: MotionClipRef,
    robot_profile: "RobotProfile",
    fps_override: float | None = None,
) -> MotionClip:
    importer = IMPORTER_BY_NAME[clip_ref.format_name]
    return importer.load_motion_clip(clip_ref, robot_profile=robot_profile, fps_override=fps_override)
