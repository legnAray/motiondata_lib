from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from motiondata_lib.robot_profiles import RobotProfile
from motiondata_lib.types import MotionClip
from motiondata_lib.viewer_defaults import DEFAULT_SCENE_PRESET


def prepare_runtime_urdf(urdf_path: Path, root_body: str) -> Path:
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

    if root.find(f"./link[@name='{root_body}']") is None:
        raise ValueError(f"Body '{root_body}' does not exist in {urdf_path}")

    if root.find("./joint[@name='floating_base_joint']") is None:
        floating_joint = ET.Element("joint", {"name": "floating_base_joint", "type": "floating"})
        ET.SubElement(floating_joint, "parent", {"link": "world"})
        ET.SubElement(floating_joint, "child", {"link": root_body})
        root.insert(1, floating_joint)

    ET.indent(tree)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix=f"{urdf_path.stem}_runtime_",
        dir=urdf_path.parent,
        delete=False,
    ) as handle:
        tree.write(handle, encoding="unicode")
        return Path(handle.name)


def apply_default_viewer_scene(spec: mujoco.MjSpec) -> None:
    scene = DEFAULT_SCENE_PRESET
    spec.visual.headlight.active = 1
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.85, 0.85, 0.85]
    spec.visual.headlight.specular = [0.20, 0.20, 0.20]

    skybox = spec.add_texture()
    skybox.name = "viewer_skybox"
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox.rgb1 = list(scene.skybox_rgb1)
    skybox.rgb2 = list(scene.skybox_rgb2)
    skybox.width = 512
    skybox.height = 3072

    floor_texture = spec.add_texture()
    floor_texture.name = "viewer_floor_texture"
    floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_texture.rgb1 = list(scene.floor_rgb1)
    floor_texture.rgb2 = list(scene.floor_rgb2)
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
    light.pos = list(scene.light_pos)
    light.dir = list(scene.light_dir)
    light.ambient = [0.15, 0.15, 0.15]
    light.diffuse = [0.85, 0.85, 0.85]
    light.specular = [0.20, 0.20, 0.20]
    light.castshadow = True


def load_model(robot_profile: RobotProfile, model_path: Path | None = None) -> mujoco.MjModel:
    urdf_path = robot_profile.model_path if model_path is None else model_path.resolve()
    runtime_urdf = prepare_runtime_urdf(urdf_path, robot_profile.root_body)
    try:
        spec = mujoco.MjSpec.from_file(str(runtime_urdf))
        apply_default_viewer_scene(spec)
        return spec.compile()
    finally:
        runtime_urdf.unlink(missing_ok=True)


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
