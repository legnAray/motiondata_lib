from __future__ import annotations

import mujoco
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from motiondata_lib.robot_profiles import RobotProfile
from motiondata_lib.viewer_defaults import DEFAULT_CAMERA_PRESET


class MujocoViewer(QOpenGLWidget):
    def __init__(self, model: mujoco.MjModel, robot_profile: RobotProfile, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.robot_profile = robot_profile
        self.data = mujoco.MjData(model)
        self.current_qpos = np.array(model.qpos0, copy=True)

        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_profile.root_body)
        self.root_body_id = root_body_id if root_body_id != -1 else None
        self.follow_root = True

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        mujoco.mjv_defaultOption(self.option)
        self.camera.distance = DEFAULT_CAMERA_PRESET.distance
        self.camera.azimuth = DEFAULT_CAMERA_PRESET.azimuth
        self.camera.elevation = DEFAULT_CAMERA_PRESET.elevation
        self.camera.lookat[:] = DEFAULT_CAMERA_PRESET.lookat

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
        self.update()

    def set_follow_root(self, enabled: bool) -> None:
        self.follow_root = enabled
        self._sync_data()
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        position = event.position()
        self._last_mouse_position = (position.x(), position.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.scene is None or self._last_mouse_position is None:
            return super().mouseMoveEvent(event)

        position = event.position()
        dx = position.x() - self._last_mouse_position[0]
        dy = position.y() - self._last_mouse_position[1]
        self._last_mouse_position = (position.x(), position.y())

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
