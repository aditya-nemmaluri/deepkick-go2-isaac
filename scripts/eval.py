import argparse
import os
import torch
import numpy as np

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

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
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))

def get_ball_relative_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ball"].data.root_pos_w - env.scene["robot"].data.root_pos_w

def is_robot_fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.scene["robot"].data.root_pos_w[:, 2] < 0.2) | (env.scene["robot"].data.root_pos_w[:, 2] > 1.2)

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
    pass

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
    base_contact = DoneTerm(func=is_robot_fallen)

@configclass
class Go2BallKickEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2BallKickSceneCfg = Go2BallKickSceneCfg(num_envs=32, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    episode_length_s: float = 20.0
    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

def main():
    env_cfg = Go2BallKickEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
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

    log_dir = os.path.expanduser("~/IsaacLab/logs/rsl_rl/go2_ball_kick")
    runner = OnPolicyRunner(vec_env, agent_cfg_dict, log_dir=log_dir, device="cuda:0")
    
    import glob
    checkpoints = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
    checkpoint_path = checkpoints[-1]
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = vec_env.reset()
    max_ball_velocities = []
    ball_displacements = []

    print("[INFO] Running headless evaluation metrics...")
    for step in range(200):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)

        ball_vel_x = env.scene["ball"].data.root_lin_vel_w[:, 0].cpu().numpy()
        ball_pos_x = env.scene["ball"].data.root_pos_w[:, 0].cpu().numpy()
        
        max_ball_velocities.append(np.max(ball_vel_x))
        ball_displacements.append(np.mean(ball_pos_x - 0.6))

    print("\n" + "="*50)
    print("      HEADLESS POLICY PERFORMANCE SUMMARY      ")
    print("="*50)
    print(f"Checkpoint Loaded            : {os.path.basename(checkpoint_path)}")
    print(f"Peak Ball Forward Velocity   : {np.max(max_ball_velocities):.2f} m/s")
    print(f"Mean Ball Distance Moved     : {ball_displacements[-1]:.2f} meters")
    print(f"Kick Success Rate (>0.5m)    : {np.mean(np.array(ball_displacements) > 0.5) * 100:.1f}%")
    print("="*50 + "\n")

    env.close()

if __name__ == "__main__":
    main()
