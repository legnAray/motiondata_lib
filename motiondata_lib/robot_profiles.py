from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_CONFIG_DIR = REPO_ROOT / "robots"
DEFAULT_ROBOT_NAME = "unitree_g1"


@dataclass(frozen=True)
class RobotProfile:
    name: str
    display_name: str
    model_path: Path
    root_body: str
    joint_names: tuple[str, ...]
    config_path: Path


def available_robot_names(config_dir: Path = ROBOT_CONFIG_DIR) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in config_dir.glob("*.toml") if path.is_file()))


def load_robot_profile(name: str, config_dir: Path = ROBOT_CONFIG_DIR) -> RobotProfile:
    config_path = (config_dir / f"{name}.toml").resolve()
    if not config_path.is_file():
        available = ", ".join(available_robot_names(config_dir)) or "(none)"
        raise ValueError(f"Unknown robot profile '{name}'. Available profiles: {available}")

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Robot config must be a TOML table: {config_path}")

    profile_name = str(data.get("name", name))
    display_name = str(data.get("display_name", profile_name))
    root_body = str(data["root_body"])
    model_value = data["model"]
    joint_values = data["joint_names"]

    if not isinstance(model_value, str):
        raise ValueError(f"Expected 'model' to be a string in {config_path}")
    if not isinstance(joint_values, list) or not joint_values:
        raise ValueError(f"Expected 'joint_names' to be a non-empty list in {config_path}")

    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()

    return RobotProfile(
        name=profile_name,
        display_name=display_name,
        model_path=model_path,
        root_body=root_body,
        joint_names=tuple(str(value) for value in joint_values),
        config_path=config_path,
    )
