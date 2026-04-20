from __future__ import annotations

import argparse
from datetime import datetime
import shutil
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
REQUIRED_KEYS = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
RIGHT_PANEL_WIDTH = 420
ROOT_BODY_NAME = "pelvis"


@dataclass(frozen=True)
class MotionClipRef:
    path: Path
    display_name: str


@dataclass(frozen=True)
class MotionClip:
    path: Path
    display_name: str
    framerate: float
    qpos_frames: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.qpos_frames.shape[0])


def discover_motion_clips(dataset_dir: Path) -> list[MotionClipRef]:
    dataset_dir = dataset_dir.resolve()
    clips = []
    for path in sorted(dataset_dir.rglob("*.npz")):
        relative = path.relative_to(dataset_dir).with_suffix("")
        clips.append(MotionClipRef(path=path, display_name="/".join(relative.parts)))
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


def load_motion_clip(clip_ref: MotionClipRef, model: mujoco.MjModel) -> MotionClip:
    with np.load(clip_ref.path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{clip_ref.path} is missing required arrays: {missing_list}")

        framerate = float(np.asarray(data["framerate"]).item())
        joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        base_pos = np.asarray(data["base_pos_w"], dtype=np.float64)
        base_quat = np.asarray(data["base_quat_w"], dtype=np.float64)

    frame_count = joint_pos.shape[0]
    if frame_count == 0:
        raise ValueError(f"{clip_ref.path} does not contain any frames")
    if base_pos.shape != (frame_count, 3):
        raise ValueError(f"{clip_ref.path} has base_pos_w shape {base_pos.shape}, expected ({frame_count}, 3)")
    if base_quat.shape != (frame_count, 4):
        raise ValueError(
            f"{clip_ref.path} has base_quat_w shape {base_quat.shape}, expected ({frame_count}, 4)"
        )
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"{clip_ref.path} has {joint_pos.shape[1]} joint columns but {len(joint_names)} joint names"
        )

    qpos_frames = np.repeat(model.qpos0[np.newaxis, :], frame_count, axis=0)
    qpos_frames[:, :3] = base_pos

    quat_norms = np.linalg.norm(base_quat, axis=1, keepdims=True)
    qpos_frames[:, 3:7] = base_quat / np.clip(quat_norms, 1e-8, None)

    for column, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint '{joint_name}' in {clip_ref.path} does not exist in the loaded model")

        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint '{joint_name}' uses unsupported MuJoCo joint type {joint_type}")

        qpos_address = model.jnt_qposadr[joint_id]
        qpos_frames[:, qpos_address] = joint_pos[:, column]

    return MotionClip(
        path=clip_ref.path,
        display_name=clip_ref.display_name,
        framerate=framerate if framerate > 0 else 30.0,
        qpos_frames=qpos_frames,
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
    def __init__(self, dataset_dir: Path, model_path: Path) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir.resolve()
        self.model_path = model_path.resolve()
        self.model = load_model(self.model_path)
        self.viewer = MujocoViewer(self.model)

        self.clips = discover_motion_clips(self.dataset_dir)
        if not self.clips:
            raise ValueError(f"No .npz files were found under {self.dataset_dir}")

        self.current_clip: MotionClip | None = None
        self.current_frame = 0.0
        self.scrubbing = False
        self.is_playing = True
        self._last_tick = time.perf_counter()

        self.list_widget = QListWidget()
        self.clip_count_label = QLabel(f"{len(self.clips)} motion files")
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
                relative_path = clip_ref.path.relative_to(self.dataset_dir)
                target_path = export_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip_ref.path, target_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported {len(checked_refs)} motion files to:\n{export_dir}",
        )

    def _load_clip(self, clip_ref: MotionClipRef) -> None:
        clip = load_motion_clip(clip_ref, self.model)
        self.current_clip = clip
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
        if self.current_clip is None:
            return

        frame_index = min(int(self.current_frame), self.current_clip.frame_count - 1)
        self.viewer.set_qpos(self.current_clip.qpos_frames[frame_index])
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
    parser = argparse.ArgumentParser(description="Browse retargeted G1 motion clips in a simple MuJoCo GUI.")
    parser.add_argument("dataset", type=Path, help="Directory containing .npz motion clips")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"URDF model to load (default: {DEFAULT_MODEL_PATH})",
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
        window = MotionBrowserWindow(dataset_dir, model_path)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Failed to start motion browser", str(exc))
        return 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
