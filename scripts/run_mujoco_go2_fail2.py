import os
import time
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import torch
from robot_descriptions.go2_mj_description import MJCF_PATH

SLOW_MO_FACTOR = 2.0
POLICY_FREQ_HZ = 50

# 1. Prepare Scene XML
model_dir = os.path.dirname(MJCF_PATH)
tree = ET.parse(MJCF_PATH)
root = tree.getroot()

for keyframe in root.findall("keyframe"):
    root.remove(keyframe)

worldbody = root.find("worldbody")
if worldbody is None:
    worldbody = ET.SubElement(root, "worldbody")

worldbody.append(ET.fromstring('<geom name="floor" size="0 0 .05" type="plane" condim="3"/>'))
worldbody.append(ET.fromstring('''
<body name="ball" pos="0.6 0 0.1">
  <freejoint/>
  <geom name="ball_geom" type="sphere" size="0.10" mass="0.2" rgba="1 0.2 0 1"/>
</body>
'''))

temp_scene_path = os.path.join(model_dir, "temp_kick_scene.xml")
tree.write(temp_scene_path)

try:
    model = mujoco.MjModel.from_xml_path(temp_scene_path)
finally:
    if os.path.exists(temp_scene_path):
        os.remove(temp_scene_path)

data = mujoco.MjData(model)

policy_path = "go2_kick_actor.pt"
if not os.path.exists(policy_path):
    policy_path = "checkpoints/go2_kick_actor.pt"

policy = torch.load(policy_path, map_location="cpu", weights_only=False)
policy.eval()

ISAAC_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
]

mujoco_joint_ids = []
for name in ISAAC_JOINTS:
    j_id = -1
    for i in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if j_name and (j_name == name or j_name == name.replace("_joint", "")):
            j_id = i
            break
    mujoco_joint_ids.append(j_id)

default_dof_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5])
action_scale = 0.25

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.35

dt = model.opt.timestep
decimation = max(1, int((1.0 / POLICY_FREQ_HZ) / dt))
target_pos = default_dof_pos.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"[INFO] Running frame-corrected policy at {POLICY_FREQ_HZ} Hz.")
    step_count = 0

    while viewer.is_running():
        start_time = time.time()

        if step_count % decimation == 0:
            # 1. Joint Positions and Velocities
            mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
            mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])
            joint_pos_rel = mj_qpos - default_dof_pos

            # 2. Get Robot Orientation Matrix (World to Base Frame)
            quat = data.qpos[3:7]  # [w, x, y, z]
            rot_mat = np.zeros((3, 3))
            mujoco.mju_quat2Mat(rot_mat.flatten(), quat)

            # 3. Rotate Velocities into Local Base Frame
            world_lin_vel = data.qvel[:3]
            world_ang_vel = data.qvel[3:6]
            base_lin_vel = rot_mat.T @ world_lin_vel
            base_ang_vel = rot_mat.T @ world_ang_vel

            # 4. Rotate Ball Relative Position into Local Base Frame
            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            world_ball_rel = data.xpos[ball_body_id] - data.qpos[:3]
            ball_rel_pos = rot_mat.T @ world_ball_rel

            # 5. Build Observation
            obs = np.concatenate([joint_pos_rel, mj_qvel, base_lin_vel, base_ang_vel, ball_rel_pos])
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                actions = policy(obs_tensor).squeeze(0).numpy()

            target_pos = default_dof_pos + action_scale * actions

        for idx, j_id in enumerate(mujoco_joint_ids):
            for act_id in range(model.nu):
                if model.actuator_trnid[act_id, 0] == j_id:
                    data.ctrl[act_id] = target_pos[idx]
                    break

        mujoco.mj_step(model, data)
        step_count += 1
        viewer.sync()

        elapsed = time.time() - start_time
        sleep_time = (dt * SLOW_MO_FACTOR) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
