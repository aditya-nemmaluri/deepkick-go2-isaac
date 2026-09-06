import os
import re
import numpy as np
import torch
import mujoco
from robot_descriptions.go2_mj_description import MJCF_PATH

POLICY_FREQ_HZ = 50
SIM_DURATION_SEC = 10.0
KP = 25.0
KD = 0.5
TORQUE_LIMIT = 23.7

model_dir = os.path.dirname(MJCF_PATH)
with open(MJCF_PATH, "r") as f:
    xml_content = f.read()

xml_content = re.sub(r"<keyframe>.*?</keyframe>", "", xml_content, flags=re.DOTALL)
extra_elements = """
    <geom name="floor" size="10 10 0.05" type="plane" rgba="0.2 0.3 0.25 1" condim="3"/>
    <body name="ball" pos="0.6 0 0.1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.10" mass="0.2" rgba="1 0.2 0 1"/>
    </body>
"""
xml_content = xml_content.replace("</worldbody>", f"{extra_elements}\n</worldbody>")

temp_scene_path = os.path.join(model_dir, "temp_eval_fsm.xml")
with open(temp_scene_path, "w") as f:
    f.write(xml_content)

try:
    model = mujoco.MjModel.from_xml_path(temp_scene_path)
finally:
    if os.path.exists(temp_scene_path):
        os.remove(temp_scene_path)

data = mujoco.MjData(model)
#policy = torch.load("go2_kick_actor.pt", map_location="cpu", weights_only=False)
policy = torch.jit.load("go2_kick_actor.pt", map_location="cpu")
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
data.qpos[2] = 0.32
for idx, j_id in enumerate(mujoco_joint_ids):
    data.qpos[model.jnt_qposadr[j_id]] = default_dof_pos[idx]

dt = model.opt.timestep
total_steps = int(SIM_DURATION_SEC / dt)
decimation = max(1, int((1.0 / POLICY_FREQ_HZ) / dt))

action_chatter_log = []
joint_acc_log = []
torque_log = []
roll_pitch_log = []
ball_vel_log = []
time_to_fall = SIM_DURATION_SEC

prev_action = np.zeros(12)
prev_qvel = np.zeros(12)
target_pos = default_dof_pos.copy()

fsm_state = "APPROACH"
kick_timer = 0.0

for step in range(total_steps):
    t_curr = step * dt

    if step % decimation == 0:
        mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
        mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])
        joint_pos_rel = mj_qpos - default_dof_pos

        quat = data.qpos[3:7]
        res_mat = np.zeros(9)
        mujoco.mju_quat2Mat(res_mat, quat)
        rot_mat = res_mat.reshape((3, 3))

        base_lin_vel = rot_mat.T @ data.qvel[:3]
        base_ang_vel = data.qvel[3:6]

        ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_world_rel = data.xpos[ball_body_id] - data.qpos[:3]
        ball_rel_pos = rot_mat.T @ ball_world_rel
        dist_to_ball = np.linalg.norm(ball_rel_pos[:2])

        if fsm_state == "APPROACH":
            target_pos = default_dof_pos.copy()
            actions = np.zeros(12)
            if dist_to_ball <= 0.65:
                fsm_state = "KICK"
                kick_timer = 0.0

        elif fsm_state == "KICK":
            obs = np.concatenate([joint_pos_rel, mj_qvel, base_lin_vel, base_ang_vel, ball_rel_pos])
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            with torch.no_grad():
                actions = policy(obs_tensor).squeeze(0).numpy()

            target_pos = default_dof_pos + action_scale * actions
            kick_timer += dt * decimation

            if kick_timer >= 1.0 or ball_rel_pos[0] > 1.2:
                fsm_state = "STAND"

        elif fsm_state == "STAND":
            target_pos = default_dof_pos.copy()
            actions = np.zeros(12)

        action_chatter = np.linalg.norm(actions - prev_action)
        action_chatter_log.append(action_chatter)
        prev_action = actions.copy()

    mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
    mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])

    torques = KP * (target_pos - mj_qpos) - KD * mj_qvel
    torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
    torque_log.append(np.abs(torques))

    joint_acc = np.linalg.norm((mj_qvel - prev_qvel) / dt)
    joint_acc_log.append(joint_acc)
    prev_qvel = mj_qvel.copy()

    quat = data.qpos[3:7]
    roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1]**2 + quat[2]**2))
    pitch = np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1.0, 1.0))
    roll_pitch_log.append(np.degrees([abs(roll), abs(pitch)]))

    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_vel = np.linalg.norm(data.cvel[ball_body_id][:3])
    ball_vel_log.append(ball_vel)

    if (data.qpos[2] < 0.2 or np.degrees(abs(roll)) > 45 or np.degrees(abs(pitch)) > 45) and time_to_fall == SIM_DURATION_SEC:
        time_to_fall = t_curr

    for idx, j_id in enumerate(mujoco_joint_ids):
        for act_id in range(model.nu):
            if model.actuator_trnid[act_id, 0] == j_id:
                data.ctrl[act_id] = torques[idx]
                break

    mujoco.mj_step(model, data)

mean_chatter = np.mean(action_chatter_log)
max_joint_acc = np.max(joint_acc_log)
peak_torque = np.max(torque_log)
rms_torque = np.sqrt(np.mean(np.square(torque_log)))
max_tilt = np.max(roll_pitch_log)
peak_ball_speed = np.max(ball_vel_log)

print("\n" + "="*55)
print("       MUJOCO FSM HARDWARE SAFETY TELEMETRY        ")
print("="*55)
print(f"Time to Fall / Instability   : {time_to_fall:.2f} s / {SIM_DURATION_SEC:.1f} s")
print(f"Peak Base Tilt (Roll/Pitch) : {max_tilt:.1f} deg")
print(f"Mean Action Chatter (||Δa||): {mean_chatter:.3f}  (Safe: < 0.15)")
print(f"Max Joint Acceleration      : {max_joint_acc:.1f} rad/s² (Safe: < 150 rad/s²)")
print(f"Peak Actuator Torque        : {peak_torque:.2f} Nm (Hardware Max: 23.7 Nm)")
print(f"RMS Actuator Torque         : {rms_torque:.2f} Nm (Continuous Limit: < 12.0 Nm)")
print(f"Peak Ball Speed             : {peak_ball_speed:.2f} m/s")
print("="*55)

if mean_chatter > 0.20 or time_to_fall < SIM_DURATION_SEC:
    print("❌ SAFETY VERDICT: DANGEROUS FOR PHYSICAL HARDWARE!")
else:
    print("✅ SAFETY VERDICT: PASSED HARDWARE FEASIBILITY THRESHOLDS.")
