# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a small Python 3.11 project with a root-level entrypoint in `main.py`. Project metadata and dependencies live in `pyproject.toml`, and `README.md` should hold user-facing usage notes. The local virtual environment is `.venv/`; treat it as machine-specific and do not commit changes from it.

If the codebase grows, move reusable logic into a package directory such as `motiondata_lib/` and mirror it with tests under `tests/`. Keep modules focused and group related motion-processing code together instead of expanding `main.py` into a catch-all script.

## Build, Test, and Development Commands
- Use `uv` for Python dependency management and execution so environments stay reproducible across machines. For example, install packages with `uv add <package>` instead of `pip install <package>`.
- `uv sync` - create or refresh the local environment from `pyproject.toml`.
- `uv run python main.py` - run the current smoke test for the app entrypoint.
- `uv run python -m compileall .` - perform a fast syntax check across the repo.
- `uv run python -m pytest` - run tests after adding `pytest` and a `tests/` directory.

There is no package build step configured yet, so validation is currently run-focused rather than artifact-focused.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep functions small, prefer standard-library solutions first, and add type hints when introducing reusable helpers or public APIs.

No formatter or linter is configured yet, so keep imports tidy and formatting consistent with the existing simple style.

## Testing Guidelines
There is no committed automated test suite yet. Add new tests under `tests/` using names like `test_parser.py` or `test_main.py`. Favor small unit tests for data transforms and one smoke test for command-line behavior when `main.py` changes.

When adding nontrivial logic, include enough tests to cover new branches and failure cases.

## Commit & Pull Request Guidelines
This repository has no shared commit history yet, so start with clear, imperative commit messages such as `feat: add frame timestamp parser` or `fix: handle empty motion sample`.

Pull requests should stay narrow and include: a short summary, linked issue or task when available, the commands used to validate the change, and sample output for behavior changes. Update `README.md` when setup or runtime behavior changes.
