import argparse
import os
import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render Go2 Ball Kick Video")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# 1. Scene Configuration with Dome Light & Camera
@configclass
class Go2BallKickSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.2, angular_damping=0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.10)),
    )
    # Ambient + Distant Lighting for Headless Vulkan
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        update_period=0.02,
        height=720,
        width=1280,
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(-1.8, 0.0, 0.9), rot=(0.965, 0.0, 0.258, 0.0), convention="world"),
    )

# 2. Helper functions
def reward_ball_forward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(env.scene["ball"].data.root_lin_vel_w[:, 0], min=0.0)

def reward_approach_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    dist = torch.norm(env.scene["ball"].data.root_pos_w[:, :2] - env.scene["robot"].data.root_pos_w[:, :2], dim=-1)
    return torch.exp(-2.0 * dist)

def get_ball_relative_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ball"].data.root_pos_w - env.scene["robot"].data.root_pos_w

def is_robot_fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.scene["robot"].data.root_pos_w[:, 2] < 0.2) | (env.scene["robot"].data.root_pos_w[:, 2] > 1.2)

# 3. MDP Configurations
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        ball_rel_pos = ObsTerm(func=get_ball_relative_position)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    approach_ball = RewTerm(func=reward_approach_ball, weight=2.0)
    kick_ball = RewTerm(func=reward_ball_forward_velocity, weight=10.0)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
    base_contact = DoneTerm(func=is_robot_fallen)

@configclass
class Go2BallKickEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2BallKickSceneCfg = Go2BallKickSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = 10.0
    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

def main():
    print("[INFO] Initializing environment and loading RTX renderer...")
    env_cfg = Go2BallKickEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)

    agent_cfg_dict = {
        "algorithm": {"class_name": "PPO"},
        "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "runner": {"class_name": "OnPolicyRunner", "num_steps_per_env": 24, "max_iterations": 1500, "save_interval": 50, "experiment_name": "go2_ball_kick", "empirical_normalization": False},
        "num_steps_per_env": 24, "max_iterations": 1500, "save_interval": 50, "experiment_name": "go2_ball_kick", "empirical_normalization": False
    }

    checkpoint_path = os.path.expanduser("~/deepkick-go2-isaac/checkpoints/model_1499.pt")
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=os.path.dirname(checkpoint_path), device="cuda:0")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = vec_env.reset()

    # Camera Warmup Steps (Warms up Vulkan textures & render pipeline)
    print("[INFO] Warming up camera buffers...")
    for _ in range(15):
        env.sim.render()
        simulation_app.update()

    frames = []
    print("[INFO] Rendering MP4 video frames offscreen...")
    for step in range(200):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)

        # Force render pass update
        env.sim.render()
        
        # Read RGB output tensor
        rgba = env.scene["camera"].data.output["rgb"][0].cpu().numpy()
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        frames.append(bgr)

        if step % 50 == 0:
            print(f"[PROGRESS] Captured frame {step}/200 (Mean Intensity: {np.mean(bgr):.1f})...")

    video_path = os.path.expanduser("~/go2_kick_demo.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 50, (1280, 720))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"[SUCCESS] Saved video rendering to: {video_path}")

    env.close()

if __name__ == "__main__":
    main()
