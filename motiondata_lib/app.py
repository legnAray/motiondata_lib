from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMessageBox

from motiondata_lib.constants import DEFAULT_MODEL_PATH
from motiondata_lib.importers import SUPPORTED_DATASET_FORMATS
from motiondata_lib.window import MotionBrowserWindow


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
