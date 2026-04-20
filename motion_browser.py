from __future__ import annotations

import argparse
import csv
from datetime import datetime
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


DEFAULT_MODEL_PATH = Path("resources/unitree_g1/g1_29dof.urdf")
STANDARD_NPZ_KEYS = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
SUPPORTED_DATASET_FORMATS = ("retargeted_npz", "sonic", "lafan1", "amass")
SUPPORTED_MOTION_SUFFIXES = {".npz", ".csv", ".npy"}
RIGHT_PANEL_WIDTH = 420
ROOT_BODY_NAME = "pelvis"
SONIC_DEFAULT_FPS = 120.0
LAFAN1_DEFAULT_FPS = 30.0
AMASS_BASE_Z_OFFSET = 0.75
CANONICAL_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
SONIC_ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
SONIC_ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")
SONIC_JOINT_COLUMNS = tuple(f"{name}_dof" for name in CANONICAL_JOINT_NAMES)
SONIC_EXPECTED_COLUMNS = ("Frame", *SONIC_ROOT_TRANSLATE_COLUMNS, *SONIC_ROOT_ROTATE_COLUMNS, *SONIC_JOINT_COLUMNS)


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


def _motion_files_under_dir(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_MOTION_SUFFIXES
    )


def _normalize_quaternions(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    return quat_wxyz / np.clip(norms, 1e-8, None)


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return _normalize_quaternions(quat_xyzw[:, [3, 0, 1, 2]])


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _euler_xyz_degrees_to_quat_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    half_angles = 0.5 * euler_rad
    sin_half = np.sin(half_angles)
    cos_half = np.cos(half_angles)

    qx = np.stack((cos_half[:, 0], sin_half[:, 0], np.zeros(len(euler_deg)), np.zeros(len(euler_deg))), axis=-1)
    qy = np.stack((cos_half[:, 1], np.zeros(len(euler_deg)), sin_half[:, 1], np.zeros(len(euler_deg))), axis=-1)
    qz = np.stack((cos_half[:, 2], np.zeros(len(euler_deg)), np.zeros(len(euler_deg)), sin_half[:, 2]), axis=-1)

    quat_wxyz = _quat_multiply(qz, _quat_multiply(qy, qx))
    return _normalize_quaternions(quat_wxyz)


def _extract_amass_fps(path: Path) -> float:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Could not infer AMASS frame rate from filename: {path.name}")
    return float(matches[-1])


def detect_motion_file_format(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if STANDARD_NPZ_KEYS.issubset(data.files):
                return "retargeted_npz"
        raise ValueError(f"Unsupported .npz layout in {path}")

    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2 and array.shape[1] == 36:
            return "amass"
        raise ValueError(f"Unsupported .npy layout in {path}: expected shape (N, 36), got {array.shape}")

    if suffix == ".csv":
        with path.open(newline="") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            raise ValueError(f"CSV file is empty: {path}")

        header = [value.strip() for value in first_line.split(",")]
        if header and header[0] == "Frame":
            if all(column in header for column in SONIC_EXPECTED_COLUMNS):
                return "sonic"
            raise ValueError(f"Unsupported Sonic CSV layout in {path}")

        if len(header) == 36:
            try:
                [float(value) for value in header]
            except ValueError as exc:
                raise ValueError(f"Unsupported CSV layout in {path}") from exc
            return "lafan1"

        raise ValueError(f"Unsupported CSV layout in {path}")

    raise ValueError(f"Unsupported motion file extension: {path.suffix}")


def detect_dataset_format(dataset_dir: Path) -> str:
    motion_files = _motion_files_under_dir(dataset_dir)
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
        if format_hint not in SUPPORTED_DATASET_FORMATS:
            raise ValueError(f"Unsupported dataset format '{format_hint}'")
        dataset_format = format_hint

    motion_files = _motion_files_under_dir(dataset_dir)
    clips = []
    for path in motion_files:
        if detect_motion_file_format(path) != dataset_format:
            continue
        relative = path.relative_to(dataset_dir).with_suffix("")
        clips.append(MotionClipRef(path=path, display_name="/".join(relative.parts), format_name=dataset_format))

    if not clips:
        raise ValueError(f"No {dataset_format} motion files were found under {dataset_dir}")
    return clips


def _prepare_runtime_urdf(urdf_path: Path) -> Path:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    compiler = root.find("mujoco/compiler")
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir")
        mesh_paths = [mesh.attrib.get("filename", "") for mesh in root.findall(".//mesh")]
        if meshdir and mesh_paths and all(path.startswith(f"{meshdir}/") for path in mesh_paths):
            del compiler.attrib["meshdir"]

    if root.find("./link[@name='world']") is None:
        root.insert(0, ET.Element("link", {"name": "world"}))

    if root.find("./joint[@name='floating_base_joint']") is None:
        floating_joint = ET.Element("joint", {"name": "floating_base_joint", "type": "floating"})
        ET.SubElement(floating_joint, "parent", {"link": "world"})
        ET.SubElement(floating_joint, "child", {"link": "pelvis"})
        root.insert(1, floating_joint)

    ET.indent(tree)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="g1_runtime_",
        dir=urdf_path.parent,
        delete=False,
    ) as handle:
        tree.write(handle, encoding="unicode")
        return Path(handle.name)


def _apply_default_viewer_scene(spec: mujoco.MjSpec) -> None:
    spec.visual.headlight.active = 1
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.85, 0.85, 0.85]
    spec.visual.headlight.specular = [0.20, 0.20, 0.20]

    skybox = spec.add_texture()
    skybox.name = "viewer_skybox"
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox.rgb1 = [0.18, 0.35, 0.55]
    skybox.rgb2 = [0.0, 0.0, 0.0]
    skybox.width = 512
    skybox.height = 3072

    floor_texture = spec.add_texture()
    floor_texture.name = "viewer_floor_texture"
    floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_texture.rgb1 = [0.10, 0.20, 0.34]
    floor_texture.rgb2 = [0.18, 0.32, 0.52]
    floor_texture.width = 512
    floor_texture.height = 512

    floor_material = spec.add_material()
    floor_material.name = "viewer_floor_material"
    floor_material.reflectance = 0.12
    floor_material.shininess = 0.05
    floor_material.specular = 0.15
    floor_material.texrepeat = [6.0, 6.0]
    floor_material.texuniform = True
    floor_material.textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    floor_material.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = floor_texture.name

    ground = spec.worldbody.add_geom()
    ground.name = "viewer_ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [10.0, 10.0, 0.1]
    ground.material = floor_material.name
    ground.pos = [0.0, 0.0, 0.0]
    ground.conaffinity = 0
    ground.contype = 0

    light = spec.worldbody.add_light()
    light.name = "viewer_key_light"
    light.pos = [0.0, 0.0, 4.5]
    light.dir = [0.0, 0.0, -1.0]
    light.ambient = [0.15, 0.15, 0.15]
    light.diffuse = [0.85, 0.85, 0.85]
    light.specular = [0.20, 0.20, 0.20]
    light.castshadow = True


def load_model(urdf_path: Path) -> mujoco.MjModel:
    runtime_urdf = _prepare_runtime_urdf(urdf_path)
    try:
        spec = mujoco.MjSpec.from_file(str(runtime_urdf))
        _apply_default_viewer_scene(spec)
        return spec.compile()
    finally:
        runtime_urdf.unlink(missing_ok=True)


def _validate_motion_clip_arrays(
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


def load_motion_clip(clip_ref: MotionClipRef, fps_override: float | None = None) -> MotionClip:
    if clip_ref.format_name == "retargeted_npz":
        with np.load(clip_ref.path, allow_pickle=False) as data:
            missing = STANDARD_NPZ_KEYS.difference(data.files)
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"{clip_ref.path} is missing required arrays: {missing_list}")

            framerate = float(np.asarray(data["framerate"]).item())
            joint_names = np.asarray(data["joint_names"])
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            base_pos_w = np.asarray(data["base_pos_w"], dtype=np.float64)
            base_quat_w = np.asarray(data["base_quat_w"], dtype=np.float64)

    elif clip_ref.format_name == "sonic":
        with clip_ref.path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = [value.strip() for value in next(reader)]

        if tuple(header[: len(SONIC_EXPECTED_COLUMNS)]) != SONIC_EXPECTED_COLUMNS:
            missing_columns = [column for column in SONIC_EXPECTED_COLUMNS if column not in header]
            raise ValueError(f"{clip_ref.path} is missing expected Sonic columns: {missing_columns}")

        column_indices = {name: index for index, name in enumerate(header)}
        use_columns = [column_indices[name] for name in SONIC_EXPECTED_COLUMNS[1:]]
        data = np.loadtxt(clip_ref.path, delimiter=",", skiprows=1, usecols=use_columns, dtype=np.float64)
        data = np.atleast_2d(data)

        joint_names = np.asarray(CANONICAL_JOINT_NAMES)
        framerate = SONIC_DEFAULT_FPS
        base_pos_w = data[:, 0:3] * 0.01
        base_quat_w = _euler_xyz_degrees_to_quat_wxyz(data[:, 3:6])
        joint_pos = np.deg2rad(data[:, 6:])

    elif clip_ref.format_name == "lafan1":
        data = np.loadtxt(clip_ref.path, delimiter=",", dtype=np.float64)
        data = np.atleast_2d(data)
        if data.shape[1] != 36:
            raise ValueError(f"{clip_ref.path} has shape {data.shape}, expected (N, 36)")

        joint_names = np.asarray(CANONICAL_JOINT_NAMES)
        framerate = LAFAN1_DEFAULT_FPS
        base_pos_w = data[:, 0:3]
        base_quat_w = _quat_xyzw_to_wxyz(data[:, 3:7])
        joint_pos = data[:, 7:]

    elif clip_ref.format_name == "amass":
        data = np.load(clip_ref.path, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] != 36:
            raise ValueError(f"{clip_ref.path} has shape {data.shape}, expected (N, 36)")

        joint_names = np.asarray(CANONICAL_JOINT_NAMES)
        framerate = _extract_amass_fps(clip_ref.path)
        base_pos_w = np.array(data[:, 0:3], copy=True)
        base_pos_w[:, 2] += AMASS_BASE_Z_OFFSET
        base_quat_w = _quat_xyzw_to_wxyz(data[:, 3:7])
        joint_pos = data[:, 7:]

    else:
        raise ValueError(f"Unsupported dataset format '{clip_ref.format_name}'")

    if fps_override is not None:
        framerate = fps_override

    joint_names = np.asarray(joint_names)
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    base_quat_w = _normalize_quaternions(np.asarray(base_quat_w, dtype=np.float64))
    _validate_motion_clip_arrays(clip_ref.path, joint_names, joint_pos, base_pos_w, base_quat_w)

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


def build_qpos_frames(clip: MotionClip, model: mujoco.MjModel) -> np.ndarray:
    qpos_frames = np.repeat(model.qpos0[np.newaxis, :], clip.frame_count, axis=0)
    qpos_frames[:, :3] = clip.base_pos_w
    qpos_frames[:, 3:7] = clip.base_quat_w

    for column, joint_name in enumerate(clip.joint_names.tolist()):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' in {clip.path} does not exist in the loaded model")

        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint '{joint_name}' uses unsupported MuJoCo joint type {joint_type}")

        qpos_address = model.jnt_qposadr[joint_id]
        qpos_frames[:, qpos_address] = clip.joint_pos[:, column]

    return qpos_frames


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


class MujocoViewer(QOpenGLWidget):
    def __init__(self, model: mujoco.MjModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = mujoco.MjData(model)
        self.current_qpos = np.array(model.qpos0, copy=True)
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
        self.root_body_id = root_body_id if root_body_id != -1 else None
        self.follow_root = True

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        mujoco.mjv_defaultOption(self.option)
        self.camera.distance = 3.0
        self.camera.azimuth = 145.0
        self.camera.elevation = -20.0
        self.camera.lookat[:] = (0.0, 0.0, 0.9)

        self.scene: mujoco.MjvScene | None = None
        self.context: mujoco.MjrContext | None = None
        self._last_mouse_position: tuple[float, float] | None = None

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(960, 640)

    def initializeGL(self) -> None:
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def _sync_data(self) -> None:
        self.data.qpos[:] = self.current_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.follow_root and self.root_body_id is not None:
            self.camera.lookat[:] = self.data.xpos[self.root_body_id]

    def paintGL(self) -> None:
        if self.scene is None or self.context is None:
            return

        self._sync_data()
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        scale = self.devicePixelRatioF()
        viewport = mujoco.MjrRect(0, 0, int(self.width() * scale), int(self.height() * scale))
        mujoco.mjr_render(viewport, self.scene, self.context)

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.current_qpos[:] = qpos
        self._sync_data()
        if self.follow_root:
            self.update()
            return
        self.update()

    def set_follow_root(self, enabled: bool) -> None:
        self.follow_root = enabled
        self._sync_data()
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        pos = event.position()
        self._last_mouse_position = (pos.x(), pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.scene is None or self._last_mouse_position is None:
            return super().mouseMoveEvent(event)

        pos = event.position()
        dx = pos.x() - self._last_mouse_position[0]
        dy = pos.y() - self._last_mouse_position[1]
        self._last_mouse_position = (pos.x(), pos.y())

        if event.buttons() & Qt.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif event.buttons() & Qt.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return super().mouseMoveEvent(event)

        scale = max(self.height(), 1)
        mujoco.mjv_moveCamera(self.model, action, dx / scale, dy / scale, self.scene, self.camera)
        self.update()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self.scene is None:
            return super().wheelEvent(event)

        delta = event.angleDelta().y()
        if delta:
            zoom_step = -0.08 if delta > 0 else 0.08
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0,
                zoom_step,
                self.scene,
                self.camera,
            )
            self.update()
        super().wheelEvent(event)


class MotionBrowserWindow(QMainWindow):
    def __init__(
        self,
        dataset_dir: Path,
        model_path: Path,
        dataset_format: str = "auto",
        fps_override: float | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir.resolve()
        self.model_path = model_path.resolve()
        self.dataset_format = dataset_format
        self.fps_override = fps_override
        self.model = load_model(self.model_path)
        self.viewer = MujocoViewer(self.model)

        self.clips = discover_motion_clips(self.dataset_dir, format_hint=self.dataset_format)
        if not self.clips:
            raise ValueError(f"No motion files were found under {self.dataset_dir}")
        self.detected_format = self.clips[0].format_name

        self.current_clip: MotionClip | None = None
        self.current_qpos_frames: np.ndarray | None = None
        self.current_frame = 0.0
        self.scrubbing = False
        self.is_playing = True
        self._last_tick = time.perf_counter()

        self.list_widget = QListWidget()
        self.clip_count_label = QLabel(f"{len(self.clips)} motion files ({self.detected_format})")
        self.checked_count_label = QLabel("0 checked")
        self.dataset_field = QLineEdit(str(self.dataset_dir))
        self.current_clip_field = QLineEdit("No clip loaded")
        self.frame_label = QLabel("Frame 0 / 0")

        self.play_button = QPushButton("Pause")
        self.export_button = QPushButton("Export checked")
        self.follow_root_checkbox = QCheckBox("Follow root")
        self.speed_spin = QDoubleSpinBox()
        self.frame_slider = QSlider(Qt.Horizontal)

        self._build_ui()
        self._connect_signals()
        self._populate_clip_list()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._advance_playback)
        self.timer.start()

        self.setWindowTitle("G1 Motion Browser")
        self.resize(1440, 900)

        self.list_widget.setCurrentRow(0)

    def _build_ui(self) -> None:
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)

        self.frame_slider.setRange(0, 0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(10)
        self.frame_slider.setTracking(True)

        self.list_widget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setTextElideMode(Qt.ElideNone)

        self.dataset_field.setReadOnly(True)
        self.dataset_field.setCursorPosition(0)
        self.dataset_field.setToolTip(str(self.dataset_dir))

        self.current_clip_field.setReadOnly(True)
        self.current_clip_field.setCursorPosition(0)

        self.export_button.setEnabled(False)

        self.follow_root_checkbox.setChecked(True)
        self.follow_root_checkbox.setEnabled(self.viewer.root_body_id is not None)
        if self.viewer.root_body_id is None:
            self.follow_root_checkbox.setToolTip(f"Body '{ROOT_BODY_NAME}' was not found in the loaded model")

        right_panel = QWidget()
        right_panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        title_label = QLabel("Dataset")

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.play_button)
        controls_row.addWidget(self.follow_root_checkbox)
        controls_row.addStretch(1)
        controls_row.addWidget(QLabel("Speed"))
        controls_row.addWidget(self.speed_spin)

        right_layout.addWidget(title_label)
        right_layout.addWidget(self.dataset_field)
        right_layout.addWidget(self.clip_count_label)
        right_layout.addWidget(self.checked_count_label)
        right_layout.addWidget(self.list_widget, stretch=1)
        right_layout.addWidget(self.current_clip_field)
        right_layout.addWidget(self.export_button)
        right_layout.addLayout(controls_row)
        right_layout.addWidget(self.frame_slider)
        right_layout.addWidget(self.frame_label)

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.viewer, stretch=1)
        central_layout.addWidget(right_panel)
        self.setCentralWidget(central_widget)

    def _connect_signals(self) -> None:
        self.list_widget.currentItemChanged.connect(self._on_clip_selected)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.play_button.clicked.connect(self._toggle_playback)
        self.export_button.clicked.connect(self._export_checked_clips)
        self.follow_root_checkbox.toggled.connect(self.viewer.set_follow_root)
        self.speed_spin.valueChanged.connect(self._reset_tick_clock)
        self.frame_slider.sliderPressed.connect(self._on_scrub_started)
        self.frame_slider.sliderReleased.connect(self._on_scrub_finished)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

    def _populate_clip_list(self) -> None:
        for clip_ref in self.clips:
            item = QListWidgetItem(clip_ref.display_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, clip_ref)
            self.list_widget.addItem(item)
        self._update_checked_count()

    def _checked_clip_refs(self) -> list[MotionClipRef]:
        checked_refs: list[MotionClipRef] = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                checked_refs.append(item.data(Qt.UserRole))
        return checked_refs

    def _update_checked_count(self) -> None:
        checked_count = len(self._checked_clip_refs())
        self.checked_count_label.setText(f"{checked_count} checked")
        self.export_button.setEnabled(checked_count > 0)

    def _create_export_directory(self, destination_root: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.dataset_dir.name}_export_{timestamp}"
        export_dir = destination_root / base_name
        suffix = 1
        while export_dir.exists():
            export_dir = destination_root / f"{base_name}_{suffix}"
            suffix += 1
        export_dir.mkdir(parents=True, exist_ok=False)
        return export_dir

    def _export_checked_clips(self) -> None:
        checked_refs = self._checked_clip_refs()
        if not checked_refs:
            QMessageBox.information(self, "No motion files selected", "Check one or more motion files before exporting.")
            return

        destination = QFileDialog.getExistingDirectory(self, "Select export destination", str(self.dataset_dir.parent))
        if not destination:
            return

        destination_root = Path(destination)

        try:
            export_dir = self._create_export_directory(destination_root)
            for clip_ref in checked_refs:
                clip = load_motion_clip(clip_ref, fps_override=self.fps_override)
                relative_path = clip_ref.path.relative_to(self.dataset_dir).with_suffix(".npz")
                target_path = export_dir / relative_path
                export_motion_clip_npz(clip, target_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported {len(checked_refs)} motion files to:\n{export_dir}",
        )

    def _load_clip(self, clip_ref: MotionClipRef) -> None:
        clip = load_motion_clip(clip_ref, fps_override=self.fps_override)
        self.current_clip = clip
        self.current_qpos_frames = build_qpos_frames(clip, self.model)
        self.current_frame = 0.0
        self._last_tick = time.perf_counter()

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, clip.frame_count - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        clip_summary = f"{clip.display_name} | {clip.frame_count} frames @ {clip.framerate:.2f} fps"
        self.current_clip_field.setText(clip_summary)
        self.current_clip_field.setCursorPosition(0)
        self.current_clip_field.setToolTip(clip_summary)
        self._render_current_frame(update_slider=False)

    def _render_current_frame(self, *, update_slider: bool = True) -> None:
        if self.current_clip is None or self.current_qpos_frames is None:
            return

        frame_index = min(int(self.current_frame), self.current_clip.frame_count - 1)
        self.viewer.set_qpos(self.current_qpos_frames[frame_index])
        self.frame_label.setText(f"Frame {frame_index + 1} / {self.current_clip.frame_count}")

        if update_slider:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)

    def _advance_playback(self) -> None:
        now = time.perf_counter()
        delta = now - self._last_tick
        self._last_tick = now

        if not self.is_playing or self.scrubbing or self.current_clip is None:
            return

        frame_step = delta * self.current_clip.framerate * self.speed_spin.value()
        self.current_frame = (self.current_frame + frame_step) % self.current_clip.frame_count
        self._render_current_frame()

    def _toggle_playback(self) -> None:
        self.is_playing = not self.is_playing
        self.play_button.setText("Pause" if self.is_playing else "Play")
        self._reset_tick_clock()

    def _reset_tick_clock(self) -> None:
        self._last_tick = time.perf_counter()

    def _on_clip_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return

        clip_ref = current.data(Qt.UserRole)
        try:
            self._load_clip(clip_ref)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Failed to load motion clip", str(exc))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        del item
        self._update_checked_count()

    def _on_scrub_started(self) -> None:
        self.scrubbing = True

    def _on_scrub_finished(self) -> None:
        self.scrubbing = False
        self._reset_tick_clock()

    def _on_slider_changed(self, value: int) -> None:
        if self.current_clip is None:
            return

        self.current_frame = float(value)
        self._render_current_frame(update_slider=False)
        if not self.scrubbing:
            self._reset_tick_clock()


def configure_opengl() -> None:
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.OpenGL)
    surface_format.setProfile(QSurfaceFormat.CompatibilityProfile)
    surface_format.setVersion(2, 1)
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(surface_format)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse motion clips in a simple MuJoCo GUI.")
    parser.add_argument("dataset", type=Path, help="Directory containing motion clips in one supported format")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"URDF model to load (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--format",
        choices=("auto", *SUPPORTED_DATASET_FORMATS),
        default="auto",
        help="Dataset format override. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--fps-override",
        type=float,
        help="Override the source frame rate for every clip in the input directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset.resolve()
    model_path = args.model.resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    if not model_path.is_file():
        raise SystemExit(f"Model file does not exist: {model_path}")

    configure_opengl()
    app = QApplication(sys.argv if argv is None else ["motion-browser", *argv])

    try:
        window = MotionBrowserWindow(
            dataset_dir,
            model_path,
            dataset_format=args.format,
            fps_override=args.fps_override,
        )
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Failed to start motion browser", str(exc))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
