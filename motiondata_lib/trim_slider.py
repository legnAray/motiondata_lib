from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class TrimSlider(QWidget):
    valueChanged = Signal(int)
    trimChanged = Signal(int, int)
    sliderPressed = Signal()
    sliderReleased = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 0
        self._value = 0
        self._trim_start = 0
        self._trim_end = 0
        self._drag_target: str | None = None

        self.setMouseTracking(True)
        self.setMinimumHeight(34)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(320, 34)

    def value(self) -> int:
        return self._value

    def trimRange(self) -> tuple[int, int]:
        return self._trim_start, self._trim_end

    def setRange(self, minimum: int, maximum: int) -> None:
        minimum = int(minimum)
        maximum = int(maximum)
        if minimum > maximum:
            minimum, maximum = maximum, minimum

        self._minimum = minimum
        self._maximum = maximum
        self._value = self._clamp_value(self._value)
        self._trim_start = self._clamp_value(self._trim_start)
        self._trim_end = self._clamp_value(self._trim_end)
        if self._trim_start > self._trim_end:
            self._trim_start = self._minimum
            self._trim_end = self._maximum
        self.update()

    def setValue(self, value: int) -> None:
        clamped = self._clamp_value(value)
        if clamped == self._value:
            return
        self._value = clamped
        self.valueChanged.emit(self._value)
        self.update()

    def setTrimRange(self, start: int, end: int) -> None:
        start = self._clamp_value(start)
        end = self._clamp_value(end)
        if start > end:
            start, end = end, start
        if start == self._trim_start and end == self._trim_end:
            return
        self._trim_start = start
        self._trim_end = end
        self.trimChanged.emit(self._trim_start, self._trim_end)
        self.update()

    def resetTrimRange(self) -> None:
        self.setTrimRange(self._minimum, self._maximum)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        groove = self._groove_rect()
        start_x = self._position_from_value(self._trim_start)
        end_x = self._position_from_value(self._trim_end)
        value_x = self._position_from_value(self._value)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#42464f"))
        painter.drawRoundedRect(groove, 3, 3)

        selection_rect = QRectF(start_x, groove.top(), max(end_x - start_x, 1.0), groove.height())
        painter.setBrush(QColor("#4f86ff"))
        painter.drawRoundedRect(selection_rect, 3, 3)

        painter.setPen(QPen(QColor("#d7dce4"), 2))
        painter.drawLine(QPointF(value_x, groove.top() - 8), QPointF(value_x, groove.bottom() + 8))

        self._draw_trim_handle(painter, start_x, groove, QColor("#ffb347"))
        self._draw_trim_handle(painter, end_x, groove, QColor("#ffb347"))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        self._drag_target = self._hit_test(event.position().x())
        self.sliderPressed.emit()
        self._update_from_position(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        x_pos = event.position().x()
        if self._drag_target is not None:
            self._update_from_position(x_pos)
            event.accept()
            return

        self.setCursor(self._cursor_for_target(self._hit_test(x_pos)))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._drag_target is not None:
            self._update_from_position(event.position().x())
            self._drag_target = None
            self.sliderReleased.emit()
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_target is None:
            self.unsetCursor()
        super().leaveEvent(event)

    def _draw_trim_handle(self, painter: QPainter, x_pos: float, groove: QRectF, color: QColor) -> None:
        top_y = groove.top() - 13
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        path = QPainterPath()
        path.moveTo(x_pos, top_y)
        path.lineTo(x_pos - 6, top_y + 8)
        path.lineTo(x_pos + 6, top_y + 8)
        path.closeSubpath()
        painter.drawPath(path)

        painter.setPen(QPen(color, 2))
        painter.drawLine(QPointF(x_pos, top_y + 8), QPointF(x_pos, groove.bottom() + 6))

    def _groove_rect(self) -> QRectF:
        handle_margin = 12.0
        return QRectF(handle_margin, self.height() / 2.0 - 3.0, max(self.width() - handle_margin * 2.0, 1.0), 6.0)

    def _position_from_value(self, value: int) -> float:
        groove = self._groove_rect()
        span = max(self._maximum - self._minimum, 1)
        ratio = (self._clamp_value(value) - self._minimum) / span
        return groove.left() + ratio * groove.width()

    def _value_from_position(self, x_pos: float) -> int:
        groove = self._groove_rect()
        if groove.width() <= 0:
            return self._minimum
        ratio = (x_pos - groove.left()) / groove.width()
        ratio = min(max(ratio, 0.0), 1.0)
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def _update_from_position(self, x_pos: float) -> None:
        target_value = self._value_from_position(x_pos)
        if self._drag_target == "trim_start":
            self.setTrimRange(min(target_value, self._trim_end), self._trim_end)
            self.setCursor(self._cursor_for_target("trim_start"))
            return
        if self._drag_target == "trim_end":
            self.setTrimRange(self._trim_start, max(target_value, self._trim_start))
            self.setCursor(self._cursor_for_target("trim_end"))
            return
        self.setValue(target_value)
        self.setCursor(self._cursor_for_target("value"))

    def _hit_test(self, x_pos: float) -> str:
        start_x = self._position_from_value(self._trim_start)
        end_x = self._position_from_value(self._trim_end)
        value_x = self._position_from_value(self._value)
        threshold = 10.0

        if abs(x_pos - start_x) <= threshold:
            return "trim_start"
        if abs(x_pos - end_x) <= threshold:
            return "trim_end"
        if abs(x_pos - value_x) <= threshold:
            return "value"
        return "value"

    def _cursor_for_target(self, target: str) -> QCursor:
        if target in {"trim_start", "trim_end"}:
            return QCursor(Qt.SizeHorCursor)
        return QCursor(Qt.PointingHandCursor)

    def _clamp_value(self, value: int) -> int:
        return min(max(int(value), self._minimum), self._maximum)
