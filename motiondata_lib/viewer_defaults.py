from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPreset:
    distance: float = 3.0
    azimuth: float = 145.0
    elevation: float = -20.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.9)


@dataclass(frozen=True)
class ScenePreset:
    skybox_rgb1: tuple[float, float, float] = (0.18, 0.35, 0.55)
    skybox_rgb2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    floor_rgb1: tuple[float, float, float] = (0.10, 0.20, 0.34)
    floor_rgb2: tuple[float, float, float] = (0.18, 0.32, 0.52)
    light_pos: tuple[float, float, float] = (0.0, 0.0, 4.5)
    light_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)


DEFAULT_CAMERA_PRESET = CameraPreset()
DEFAULT_SCENE_PRESET = ScenePreset()
