from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from motiondata_lib.constants import RIGHT_PANEL_WIDTH
from motiondata_lib.exporters import export_motion_clip_npz
from motiondata_lib.importers import discover_motion_clips, load_motion_clip
from motiondata_lib.model import build_qpos_frames, load_model
from motiondata_lib.robot_profiles import RobotProfile
from motiondata_lib.trim_slider import TrimSlider
from motiondata_lib.types import MotionClip, MotionClipRef
from motiondata_lib.viewer import MujocoViewer


class MotionBrowserWindow(QMainWindow):
    def __init__(
        self,
        dataset_dir: Path,
        robot_profile: RobotProfile,
        dataset_format: str = "auto",
        model_override: Path | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir.resolve()
        self.robot_profile = robot_profile
        self.model_path = (robot_profile.model_path if model_override is None else model_override.resolve())
        self.dataset_format = dataset_format

        self.model: mujoco.MjModel = load_model(self.robot_profile, self.model_path)
        self.viewer = MujocoViewer(self.model, self.robot_profile)

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
        self.dataset_title_label = QLabel(self.dataset_dir.name or str(self.dataset_dir))
        self.dataset_meta_label = QLabel(f"{self.robot_profile.display_name} · {self.detected_format}")
        self.clip_count_label = QLabel()
        self.checked_count_label = QLabel("0 checked")
        self.dataset_field = QLineEdit(str(self.dataset_dir))
        self.filter_field = QLineEdit()
        self.current_clip_field = QLineEdit("No clip loaded")
        self.frame_label = QLabel("Frame 0 / 0")
        self.trim_label = QLabel("Trim 0 - 0 (0 frames)")

        self.play_button = QPushButton("Pause")
        self.export_button = QPushButton("Export checked")
        self.trim_button = QPushButton("Trim current")
        self.follow_root_checkbox = QCheckBox("Follow root")
        self.speed_spin = QDoubleSpinBox()
        self.frame_slider = TrimSlider()

        self._build_ui()
        self._connect_signals()
        self._populate_clip_list()

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._advance_playback)
        self.timer.start()

        self.setWindowTitle(f"{self.robot_profile.display_name} Motion Browser")
        self.resize(1440, 900)
        self.list_widget.setCurrentRow(0)

    def _build_ui(self) -> None:
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)

        self.frame_slider.setRange(0, 0)
        self.frame_slider.setTrimRange(0, 0)

        self.list_widget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setTextElideMode(Qt.ElideNone)
        self.list_widget.setSpacing(0)

        self.dataset_field.setReadOnly(True)
        self.dataset_field.setCursorPosition(0)
        self.dataset_field.setToolTip(str(self.dataset_dir))
        self.dataset_field.setObjectName("datasetField")

        self.filter_field.setPlaceholderText("Filter motions...")
        self.filter_field.setClearButtonEnabled(True)
        self.filter_field.setObjectName("filterField")

        self.current_clip_field.setReadOnly(True)
        self.current_clip_field.setCursorPosition(0)
        self.current_clip_field.setObjectName("summaryField")

        self.export_button.setEnabled(False)
        self.trim_button.setEnabled(False)
        self.export_button.setObjectName("primaryButton")

        self.follow_root_checkbox.setChecked(True)
        self.follow_root_checkbox.setEnabled(self.viewer.root_body_id is not None)
        if self.viewer.root_body_id is None:
            self.follow_root_checkbox.setToolTip(
                f"Body '{self.robot_profile.root_body}' was not found in the loaded model"
            )

        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        controls_row.addWidget(self.play_button)
        controls_row.addWidget(self.follow_root_checkbox)
        controls_row.addStretch(1)
        speed_label = QLabel("Speed")
        speed_label.setObjectName("inlineLabel")
        controls_row.addWidget(speed_label)
        controls_row.addWidget(self.speed_spin)

        library_stats_row = QHBoxLayout()
        library_stats_row.setSpacing(10)
        library_stats_row.addWidget(self.clip_count_label)
        library_stats_row.addStretch(1)
        library_stats_row.addWidget(self.checked_count_label)

        timeline_status_row = QHBoxLayout()
        timeline_status_row.setSpacing(10)
        timeline_status_row.addWidget(self.frame_label)
        timeline_status_row.addStretch(1)
        timeline_status_row.addWidget(self.trim_label)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        action_row.addWidget(self.trim_button)
        action_row.addWidget(self.export_button)

        self.dataset_title_label.setObjectName("panelTitle")
        self.dataset_meta_label.setObjectName("panelMeta")
        self.clip_count_label.setObjectName("statLabel")
        self.checked_count_label.setObjectName("statLabel")
        self.frame_label.setObjectName("statLabel")
        self.trim_label.setObjectName("statLabel")

        right_layout.addWidget(self._section_label("Dataset"))
        right_layout.addWidget(self.dataset_title_label)
        right_layout.addWidget(self.dataset_meta_label)
        right_layout.addWidget(self.dataset_field)
        right_layout.addWidget(self._section_label("Current Clip"))
        right_layout.addWidget(self.current_clip_field)
        right_layout.addWidget(self._divider())
        right_layout.addWidget(self._section_label("Motion Files"))
        right_layout.addWidget(self.filter_field)
        right_layout.addLayout(library_stats_row)
        right_layout.addWidget(self.list_widget, stretch=1)
        right_layout.addWidget(self._divider())
        right_layout.addWidget(self._section_label("Timeline"))
        right_layout.addLayout(controls_row)
        right_layout.addWidget(self.frame_slider)
        right_layout.addLayout(timeline_status_row)
        right_layout.addLayout(action_row)

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.viewer, stretch=1)
        central_layout.addWidget(right_panel)
        self.setCentralWidget(central_widget)
        self._apply_styles()

    def _connect_signals(self) -> None:
        self.list_widget.currentItemChanged.connect(self._on_clip_selected)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.filter_field.textChanged.connect(self._apply_clip_filter)
        self.play_button.clicked.connect(self._toggle_playback)
        self.export_button.clicked.connect(self._export_checked_clips)
        self.trim_button.clicked.connect(self._trim_current_clip)
        self.follow_root_checkbox.toggled.connect(self.viewer.set_follow_root)
        self.speed_spin.valueChanged.connect(self._reset_tick_clock)
        self.frame_slider.sliderPressed.connect(self._on_slider_interaction_started)
        self.frame_slider.sliderReleased.connect(self._on_slider_interaction_finished)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_slider.trimChanged.connect(self._on_trim_changed)

    def _populate_clip_list(self) -> None:
        for clip_ref in self.clips:
            item = QListWidgetItem(clip_ref.display_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, clip_ref)
            self._refresh_item_visual(item)
            self.list_widget.addItem(item)
        self._apply_clip_filter(self.filter_field.text())
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
        self._update_clip_count_label()

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
                clip = load_motion_clip(clip_ref, self.robot_profile)
                relative_path = clip_ref.path.relative_to(self.dataset_dir).with_suffix(".npz")
                export_motion_clip_npz(clip, export_dir / relative_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported {len(checked_refs)} motion files to:\n{export_dir}",
        )

    def _load_clip(self, clip_ref: MotionClipRef) -> None:
        clip = load_motion_clip(clip_ref, self.robot_profile)
        self.current_clip = clip
        self.current_qpos_frames = build_qpos_frames(clip, self.model)
        self.current_frame = 0.0
        self._last_tick = time.perf_counter()
        self.trim_button.setEnabled(True)
        self.frame_slider.setRange(0, clip.frame_count - 1)
        self.frame_slider.resetTrimRange()
        self.frame_slider.setValue(0)

        clip_summary = f"{clip.display_name} | {clip.frame_count} frames @ {clip.framerate:.2f} fps"
        self.current_clip_field.setText(clip_summary)
        self.current_clip_field.setCursorPosition(0)
        self.current_clip_field.setToolTip(clip_summary)
        self._update_trim_label()
        self._render_current_frame(update_slider=False)

    def _render_current_frame(self, *, update_slider: bool = True) -> None:
        if self.current_clip is None or self.current_qpos_frames is None:
            return

        frame_index = min(int(self.current_frame), self.current_clip.frame_count - 1)
        self.viewer.set_qpos(self.current_qpos_frames[frame_index])
        trim_start, trim_end = self.frame_slider.trimRange()
        trim_frame_count = trim_end - trim_start + 1
        self.frame_label.setText(f"Frame {frame_index + 1} / {self.current_clip.frame_count}")
        self.trim_label.setText(f"Trim {trim_start + 1} - {trim_end + 1} ({trim_frame_count} frames)")

        if update_slider:
            self.frame_slider.setValue(frame_index)

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
        self._refresh_item_visual(item)
        self._update_checked_count()

    def _on_slider_interaction_started(self) -> None:
        self.scrubbing = True

    def _on_slider_interaction_finished(self) -> None:
        self.scrubbing = False
        self._reset_tick_clock()

    def _on_slider_changed(self, value: int) -> None:
        if self.current_clip is None:
            return

        self.current_frame = float(value)
        self._render_current_frame(update_slider=False)
        if not self.scrubbing:
            self._reset_tick_clock()

    def _on_trim_changed(self, start: int, end: int) -> None:
        del start, end
        self._update_trim_label()

    def _update_trim_label(self) -> None:
        if self.current_clip is None:
            self.frame_label.setText("Frame 0 / 0")
            self.trim_label.setText("Trim 0 - 0 (0 frames)")
            return

        trim_start, trim_end = self.frame_slider.trimRange()
        trim_frame_count = trim_end - trim_start + 1
        current_index = min(int(self.current_frame), self.current_clip.frame_count - 1)
        self.frame_label.setText(f"Frame {current_index + 1} / {self.current_clip.frame_count}")
        self.trim_label.setText(f"Trim {trim_start + 1} - {trim_end + 1} ({trim_frame_count} frames)")

    def _build_trimmed_clip(self, start_frame: int, end_frame: int) -> MotionClip:
        if self.current_clip is None:
            raise ValueError("No motion clip is loaded")

        frame_slice = slice(start_frame, end_frame + 1)
        return MotionClip(
            path=self.current_clip.path,
            display_name=self.current_clip.display_name,
            format_name=self.current_clip.format_name,
            framerate=self.current_clip.framerate,
            joint_names=np.array(self.current_clip.joint_names, copy=True),
            joint_pos=np.array(self.current_clip.joint_pos[frame_slice], copy=True),
            base_pos_w=np.array(self.current_clip.base_pos_w[frame_slice], copy=True),
            base_quat_w=np.array(self.current_clip.base_quat_w[frame_slice], copy=True),
        )

    def _trim_current_clip(self) -> None:
        if self.current_clip is None:
            QMessageBox.information(self, "No motion clip loaded", "Load a motion clip before trimming.")
            return

        trim_start, trim_end = self.frame_slider.trimRange()
        trimmed_clip = self._build_trimmed_clip(trim_start, trim_end)
        suggested_name = (
            f"{self.current_clip.display_name.replace('/', '__')}_trim_{trim_start + 1}_{trim_end + 1}.npz"
        )
        default_path = str((self.dataset_dir / suggested_name).resolve())
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Save trimmed motion clip",
            default_path,
            "NumPy archive (*.npz)",
        )
        if not destination:
            return

        output_path = Path(destination)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        try:
            export_motion_clip_npz(trimmed_clip, output_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Trim export failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Trim export complete",
            f"Exported frames {trim_start + 1} - {trim_end + 1} to:\n{output_path}",
        )

    def _apply_clip_filter(self, text: str) -> None:
        filter_text = text.strip().lower()
        first_visible: QListWidgetItem | None = None
        current_item = self.list_widget.currentItem()

        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            visible = filter_text in item.text().lower()
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item

        if current_item is not None and current_item.isHidden() and first_visible is not None:
            self.list_widget.setCurrentItem(first_visible)

        self._update_clip_count_label()

    def _update_clip_count_label(self) -> None:
        total_count = len(self.clips)
        visible_count = sum(not self.list_widget.item(index).isHidden() for index in range(self.list_widget.count()))
        if visible_count == total_count:
            self.clip_count_label.setText(f"{total_count} motions")
            return
        self.clip_count_label.setText(f"{visible_count} / {total_count} motions")

    def _refresh_item_visual(self, item: QListWidgetItem) -> None:
        font = item.font()
        is_checked = item.checkState() == Qt.Checked
        font.setBold(is_checked)
        item.setFont(font)
        item.setForeground(QColor("#f3f6fb") if is_checked else QColor("#d6dde8"))
        item.setBackground(QColor("#162131") if is_checked else QColor("#00000000"))

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _divider(self) -> QFrame:
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        return divider

    def _apply_styles(self) -> None:
        assets_dir = Path(__file__).resolve().parent / "assets"
        checkmark_icon = (assets_dir / "checkmark.svg").as_posix()
        spin_up_icon = (assets_dir / "spin_up.svg").as_posix()
        spin_down_icon = (assets_dir / "spin_down.svg").as_posix()
        self.setStyleSheet(
            """
            QMainWindow, QWidget#centralWidget {
                background: #101419;
                color: #e8edf5;
            }
            QWidget#rightPanel {
                background: #171c24;
                border-left: 1px solid #28303a;
            }
            QLabel {
                color: #d6dde8;
            }
            QLabel#panelTitle {
                color: #f5f7fb;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#panelMeta,
            QLabel#statLabel,
            QLabel#inlineLabel,
            QLabel#sectionTitle {
                color: #8e9bb0;
            }
            QLabel#sectionTitle {
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            QLineEdit,
            QDoubleSpinBox {
                background: #10151b;
                color: #eef3fa;
                border: 1px solid #2b3440;
                border-radius: 6px;
                padding: 7px 9px;
                selection-background-color: #3d71ff;
            }
            QLineEdit:focus,
            QDoubleSpinBox:focus {
                border: 1px solid #4e7dff;
            }
            QLineEdit#datasetField,
            QLineEdit#summaryField {
                font-weight: 500;
            }
            QListWidget {
                background: #10151b;
                color: #d6dde8;
                border: 1px solid #2b3440;
                border-radius: 8px;
                outline: none;
                padding: 4px;
            }
            QListWidget::item {
                padding: 7px 10px;
                margin: 1px 0;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #315ee8;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background: #182130;
            }
            QListWidget::indicator,
            QAbstractItemView::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #5b6676;
                background: #0d1218;
            }
            QListWidget::indicator:checked,
            QAbstractItemView::indicator:checked,
            QCheckBox::indicator:checked {
                border: 1px solid #4d7aff;
                background: #142035;
                image: url(%s);
            }
            QListWidget::indicator:hover,
            QAbstractItemView::indicator:hover,
            QCheckBox::indicator:hover {
                border-color: #6f7d92;
            }
            QPushButton {
                background: #232c37;
                color: #edf2fb;
                border: 1px solid #364150;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background: #2a3441;
            }
            QPushButton:pressed {
                background: #1b232c;
            }
            QPushButton#primaryButton {
                background: #315ee8;
                border-color: #4d7aff;
            }
            QPushButton#primaryButton:hover {
                background: #3a67ef;
            }
            QPushButton:disabled {
                background: #1a2028;
                color: #697587;
                border-color: #28303a;
            }
            QPushButton#primaryButton:disabled {
                background: #223153;
                color: #7b88a0;
                border-color: #2f3d5e;
            }
            QCheckBox {
                color: #dbe2ec;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #5b6676;
                background: #0d1218;
            }
            QFrame#divider {
                background: #262e38;
                max-height: 1px;
                min-height: 1px;
                border: none;
            }
            QScrollBar:vertical,
            QScrollBar:horizontal {
                background: #12171e;
                border: 1px solid #27303a;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                width: 10px;
                margin: 4px 0 4px 0;
            }
            QScrollBar:horizontal {
                height: 10px;
                margin: 0 4px 4px 4px;
            }
            QScrollBar::handle:vertical {
                background: #2c3440;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #2c3440;
                min-width: 24px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #394352;
            }
            QScrollBar::handle:horizontal:hover {
                background: #394352;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
            QDoubleSpinBox {
                padding-right: 24px;
            }
            QDoubleSpinBox::up-button,
            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                width: 18px;
                border-left: 1px solid #2b3440;
                background: #171d24;
            }
            QDoubleSpinBox::up-button {
                subcontrol-position: top right;
                border-top-right-radius: 6px;
                border-bottom: 1px solid #2b3440;
            }
            QDoubleSpinBox::down-button {
                subcontrol-position: bottom right;
                border-bottom-right-radius: 6px;
            }
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background: #202833;
            }
            QDoubleSpinBox::up-arrow {
                image: url(%s);
                width: 10px;
                height: 8px;
            }
            QDoubleSpinBox::down-arrow {
                image: url(%s);
                width: 10px;
                height: 8px;
            }
            """
            % (checkmark_icon, spin_up_icon, spin_down_icon)
        )
