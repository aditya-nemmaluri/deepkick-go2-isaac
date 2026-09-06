import os
import re
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from robot_descriptions.go2_mj_description import MJCF_PATH

SLOW_MO_FACTOR = 1.0
POLICY_FREQ_HZ = 50
KP_WALK = 40.0
KD_WALK = 1.2
KP_POLICY = 25.0
KD_POLICY = 0.5
TORQUE_LIMIT = 23.7

# 1. Spawn Ball at 1.2m
model_dir = os.path.dirname(MJCF_PATH)
with open(MJCF_PATH, "r") as f:
    xml_content = f.read()

xml_content = re.sub(r"<keyframe>.*?</keyframe>", "", xml_content, flags=re.DOTALL)
extra_elements = """
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" size="10 10 0.05" type="plane" rgba="0.2 0.3 0.25 1" condim="3"/>
    <body name="ball" pos="1.2 0.0 0.10">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.10" mass="0.2" rgba="1 0.2 0 1"/>
    </body>
"""
xml_content = xml_content.replace("</worldbody>", f"{extra_elements}\n</worldbody>")

temp_scene_path = os.path.join(model_dir, "temp_fsm_scene.xml")
with open(temp_scene_path, "w") as f:
    f.write(xml_content)

try:
    model = mujoco.MjModel.from_xml_path(temp_scene_path)
finally:
    if os.path.exists(temp_scene_path):
        os.remove(temp_scene_path)

data = mujoco.MjData(model)

# 2. Load TorchScript Policy
policy = torch.jit.load("go2_kick_actor.pt", map_location="cpu")
policy.eval()

# 3. Joint Mapping
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
decimation = max(1, int((1.0 / POLICY_FREQ_HZ) / dt))

cmd_vx = 0.0
fsm_state = "MANUAL_WALK"
settle_timer = 0.0
kick_timer = 0.0
walk_time = 0.0
prev_sim_time = 0.0

def key_callback(key):
    global cmd_vx, fsm_state
    if key in (265, 87, 119):  # 'W' / Up Arrow
        cmd_vx = 1.0
        print("[CONTROL] Walking Forward Toward Ball")
    elif key in (264, 83, 115):  # 'S' / Down Arrow
        if cmd_vx > 0:
            cmd_vx = 0.0
            print("[CONTROL] Stopped")
        else:
            cmd_vx = -0.5
            print("[CONTROL] Reversing")
    elif key == 32:  # Spacebar
        cmd_vx = 0.0
        if fsm_state == "STAND":
            fsm_state = "MANUAL_WALK"
            print("[FSM] Manual walk mode re-enabled!")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("\n" + "="*60)
    print("  CALIBRATED KICK FSM ACTIVE:")
    print("    Press 'W' -> Walk -> Plant at 0.60m -> Brief Kick (0.4s) -> Stand")
    print("="*60 + "\n")
    step_count = 0

    while viewer.is_running():
        start_time = time.time()

        if data.time < prev_sim_time:
            fsm_state = "MANUAL_WALK"
            settle_timer = 0.0
            kick_timer = 0.0
            walk_time = 0.0
            cmd_vx = 0.0
            print("[FSM] Reset detected. Stopped in MANUAL_WALK mode.")
        prev_sim_time = data.time

        if step_count % decimation == 0:
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

            # --- State Machine Logic ---
            if fsm_state == "MANUAL_WALK":
                kp_curr, kd_curr = KP_WALK, KD_WALK
                target_pos = default_dof_pos.copy()

                if cmd_vx != 0.0:
                    walk_time += dt * decimation
                    freq = 2.0
                    s = np.sin(2 * np.pi * freq * walk_time)

                    if s > 0:
                        target_pos[4] = 0.8 - 0.35 * s * cmd_vx
                        target_pos[8] = -1.5 - 0.50 * s * cmd_vx
                        target_pos[7] = 0.8 - 0.35 * s * cmd_vx
                        target_pos[11] = -1.5 - 0.50 * s * cmd_vx
                        target_pos[5] = 0.8 + 0.25 * s * cmd_vx
                        target_pos[9] = -1.5 + 0.15 * s * cmd_vx
                        target_pos[6] = 0.8 + 0.25 * s * cmd_vx
                        target_pos[10] = -1.5 + 0.15 * s * cmd_vx
                    else:
                        sp = -s
                        target_pos[5] = 0.8 - 0.35 * sp * cmd_vx
                        target_pos[9] = -1.5 - 0.50 * sp * cmd_vx
                        target_pos[6] = 0.8 - 0.35 * sp * cmd_vx
                        target_pos[10] = -1.5 - 0.50 * sp * cmd_vx
                        target_pos[4] = 0.8 + 0.25 * sp * cmd_vx
                        target_pos[8] = -1.5 + 0.15 * sp * cmd_vx
                        target_pos[7] = 0.8 + 0.25 * sp * cmd_vx
                        target_pos[11] = -1.5 + 0.15 * sp * cmd_vx

                # Trigger at trained initial position (0.60m)
                if ball_rel_pos[0] > 0.10 and ball_rel_pos[0] <= 0.62 and abs(ball_rel_pos[1]) <= 0.25:
                    fsm_state = "SETTLE"
                    settle_timer = 0.0
                    cmd_vx = 0.0
                    print("\n[FSM] Reached target distance (0.60m)! Planting stance...")

            elif fsm_state == "SETTLE":
                kp_curr, kd_curr = KP_WALK, KD_WALK
                target_pos = default_dof_pos.copy()
                settle_timer += dt * decimation

                if settle_timer >= 0.10:  # Brief 100ms pause to stabilize base
                    fsm_state = "KICK"
                    kick_timer = 0.0
                    print("[FSM] Stance planted. Handing over to go2_kick_actor.pt for 0.40s strike...")

            elif fsm_state == "KICK":
                kp_curr, kd_curr = KP_POLICY, KD_POLICY
                obs = np.concatenate([joint_pos_rel, mj_qvel, base_lin_vel, base_ang_vel, ball_rel_pos])
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

                with torch.no_grad():
                    actions = policy(obs_tensor).squeeze(0).numpy()

                target_pos = default_dof_pos + action_scale * actions
                kick_timer += dt * decimation

                # Cut off policy immediately after contact (0.40s)
                if kick_timer >= 0.40 or ball_rel_pos[0] > 0.85:
                    fsm_state = "STAND"
                    print("[FSM] Ball launched! Transitioning to STAND state.")

            elif fsm_state == "STAND":
                kp_curr, kd_curr = KP_WALK, KD_WALK
                target_pos = default_dof_pos.copy()

        # Low-level PD Controller
        mj_qpos = np.array([data.qpos[model.jnt_qposadr[j_id]] for j_id in mujoco_joint_ids])
        mj_qvel = np.array([data.qvel[model.jnt_dofadr[j_id]] for j_id in mujoco_joint_ids])

        torques = kp_curr * (target_pos - mj_qpos) - kd_curr * mj_qvel
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
