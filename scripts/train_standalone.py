import argparse
import os
import sys

# 1. Initialize Isaac Sim AppLauncher
parser = argparse.ArgumentParser(description="Train Go2 Ball Kick Task")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports (after AppLauncher starts)
import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# 3. Scene Configuration
@configclass
class Go2BallKickSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.2,
                angular_damping=0.2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.10)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )

# 4. Custom Helper Functions
def reward_ball_forward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    ball = env.scene["ball"]
    ball_vel_x = ball.data.root_lin_vel_w[:, 0]
    return torch.clamp(ball_vel_x, min=0.0)

def reward_approach_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return torch.exp(-2.0 * dist)

def get_ball_relative_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return ball.data.root_pos_w - robot.data.root_pos_w

def is_robot_fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return (robot.data.root_pos_w[:, 2] < 0.2) | (robot.data.root_pos_w[:, 2] > 1.2)

# 5. Actions, Observations, Rewards & Terminations
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )

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

# 6. Master Environment Config
@configclass
class Go2BallKickEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2BallKickSceneCfg = Go2BallKickSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    episode_length_s: float = 20.0

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

# 7. Main Training Execution
def main():
    env_cfg = Go2BallKickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)

    # RSL-RL v2.x Configuration Dictionary Schema
    agent_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "num_steps_per_env": 24,
            "max_iterations": 1500,
            "save_interval": 50,
            "experiment_name": "go2_ball_kick",
            "empirical_normalization": False,
        },
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 50,
        "experiment_name": "go2_ball_kick",
        "empirical_normalization": False,
    }

    log_dir = os.path.expanduser("~/IsaacLab/logs/rsl_rl/go2_ball_kick")
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=log_dir, device="cuda:0")

    print("[INFO] Starting headless RL training for Go2 Ball Kick...")
    runner.learn(num_learning_iterations=1500, init_at_random_ep_len=True)

    env.close()

if __name__ == "__main__":
    main()
