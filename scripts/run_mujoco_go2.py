import os
import re
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from robot_descriptions.go2_mj_description import MJCF_PATH

SLOW_MO_FACTOR = 1.5
POLICY_FREQ_HZ = 50

KP = 25.0
KD = 0.5
TORQUE_LIMIT = 23.7

# 1. Cleanly Patch Base Go2 MJCF File
model_dir = os.path.dirname(MJCF_PATH)
with open(MJCF_PATH, "r") as f:
    xml_content = f.read()

xml_content = re.sub(r"<keyframe>.*?</keyframe>", "", xml_content, flags=re.DOTALL)

extra_elements = """
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" size="10 10 0.05" type="plane" rgba="0.2 0.3 0.25 1" condim="3"/>
    <body name="ball" pos="0.6 0 0.1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.10" mass="0.2" rgba="1 0.2 0 1"/>
    </body>
"""
xml_content = xml_content.replace("</worldbody>", f"{extra_elements}\n</worldbody>")

temp_scene_path = os.path.join(model_dir, "temp_kick_scene.xml")
with open(temp_scene_path, "w") as f:
    f.write(xml_content)

try:
    model = mujoco.MjModel.from_xml_path(temp_scene_path)
finally:
    if os.path.exists(temp_scene_path):
        os.remove(temp_scene_path)

data = mujoco.MjData(model)

# 2. Load Policy
policy_path = "go2_kick_actor.pt"
policy = torch.load(policy_path, map_location="cpu", weights_only=False)
policy.eval()

# 3. Match Joint Ordering
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

# 4. Set Initial Pose
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.32  # Ground stance height

for idx, j_id in enumerate(mujoco_joint_ids):
    data.qpos[model.jnt_qposadr[j_id]] = default_dof_pos[idx]

dt = model.opt.timestep
decimation = max(1, int((1.0 / POLICY_FREQ_HZ) / dt))
target_pos = default_dof_pos.copy()

# 5. Interactive Visualizer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"[INFO] Running MuJoCo visualizer with body-frame observation alignment...")
    step_count = 0

    while viewer.is_running():
        start_time = time.time()

        if step_count % decimation == 0:
            mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
            mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])
            joint_pos_rel = mj_qpos - default_dof_pos

            # Correct 3x3 rotation matrix extraction
            quat = data.qpos[3:7]  # [w, x, y, z]
            res_mat = np.zeros(9)
            mujoco.mju_quat2Mat(res_mat, quat)
            rot_mat = res_mat.reshape((3, 3))

            # 1) Base Linear Velocity (World -> Body frame)
            base_lin_vel = rot_mat.T @ data.qvel[:3]

            # 2) Base Angular Velocity (Natively local in MuJoCo freejoint)
            base_ang_vel = data.qvel[3:6]

            # 3) Relative Ball Position (Rotated into local Body frame)
            ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_world_rel = data.xpos[ball_body_id] - data.qpos[:3]
            ball_rel_pos = rot_mat.T @ ball_world_rel

            obs = np.concatenate([joint_pos_rel, mj_qvel, base_lin_vel, base_ang_vel, ball_rel_pos])
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                actions = policy(obs_tensor).squeeze(0).numpy()

            target_pos = default_dof_pos + action_scale * actions

        # PD Control Calculations
        mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
        mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])

        torques = KP * (target_pos - mj_qpos) - KD * mj_qvel
        torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        for idx, j_id in enumerate(mujoco_joint_ids):
            for act_id in range(model.nu):
                if model.actuator_trnid[act_id, 0] == j_id:
                    data.ctrl[act_id] = torques[idx]
                    break

        mujoco.mj_step(model, data)
        step_count += 1
        viewer.sync()

        elapsed = time.time() - start_time
        sleep_time = (dt * SLOW_MO_FACTOR) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
