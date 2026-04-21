# Repository Guidelines

## Project Structure & Module Organization
The app entrypoint is `main.py`, which launches the desktop browser through `motiondata_lib/app.py`. Core GUI and runtime code lives under `motiondata_lib/`: `window.py` wires the interface, `viewer.py` and `model.py` handle MuJoCo rendering, `trim_slider.py` provides timeline trimming, and `exporters.py` writes standardized `.npz` clips. Dataset adapters live in `motiondata_lib/importers/`, while robot definitions live in `robots/*.toml` with matching URDF assets under `robots/resources/`. Use `example_datasets/` for smoke tests and format references.

## Build, Test, and Development Commands
- `uv sync` — create or refresh the local environment from `pyproject.toml`.
- `uv run python main.py example_datasets/sonic` — launch the browser with a known sample dataset.
- `uv run python main.py --help` — verify CLI options after changing startup or robot-loading behavior.
- `uv run python -m py_compile main.py motiondata_lib/*.py motiondata_lib/importers/*.py` — fast syntax check for the app and importers.

On Ubuntu, install `libxcb-cursor0` before running the GUI. Use `uv add <package>` for new Python dependencies; do not use `pip install` directly in this repo.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and explicit type hints on public helpers. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep format-specific parsing inside importer modules rather than growing `window.py` or `app.py`. New robot-specific assumptions belong in `robots/*.toml`, not hardcoded in runtime code.

## Testing Guidelines
There is no committed `pytest` suite yet, so rely on targeted smoke tests. Validate importer changes with a sample directory from `example_datasets/`, and validate GUI changes with startup checks plus a quick manual pass on playback, trim, and export behavior. When adding nontrivial parsing or transforms, prefer small unit tests under a future `tests/` directory.

## Commit & Pull Request Guidelines
Recent history uses short imperative messages such as `feat: add motion trim`, `update readme`, and `refactor codebase`. Prefer concise subject lines with a verb-first summary; use a `feat:` or `fix:` prefix when it adds clarity. Pull requests should include the user-visible behavior change, the validation commands you ran, and updated docs when CLI, formats, or setup steps change.
