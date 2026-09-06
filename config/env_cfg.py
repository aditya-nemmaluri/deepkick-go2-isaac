import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# 1. SCENE CONFIGURATION
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

# 2. REWARD TERMS
def reward_ball_forward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    ball = env.scene["ball"]
    ball_vel_x = ball.data.root_lin_vel_w[:, 0]
    return torch.clamp(ball_vel_x, min=0.0)

def reward_approach_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    dist = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return torch.exp(-2.0 * dist)

# 3. OBSERVATIONS & REWARDS
@configclass
class ActionsCfg:
    joint_pos = SceneEntityCfg("robot")

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=lambda env: env.scene["robot"].data.joint_pos)
        joint_vel = ObsTerm(func=lambda env: env.scene["robot"].data.joint_vel)
        base_lin_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_lin_vel_b)
        base_ang_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_ang_vel_b)
        ball_rel_pos = ObsTerm(
            func=lambda env: env.scene["ball"].data.root_pos_w - env.scene["robot"].data.root_pos_w
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    approach_ball = RewTerm(func=reward_approach_ball, weight=2.0)
    kick_ball = RewTerm(func=reward_ball_forward_velocity, weight=10.0)

# 4. MASTER ENV CONFIG
@configclass
class Go2BallKickEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2BallKickSceneCfg = Go2BallKickSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

# 5. RSL-RL AGENT CONFIG
@configclass
class Go2BallKickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_ball_kick"
    empirical_normalization = False
    device = "cuda:0"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dimensions=[512, 256, 128],
        critic_hidden_dimensions=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# 6. TASK REGISTRATION
gym.register(
    id="Isaac-Go2-BallKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "cfg_entry_point": Go2BallKickEnvCfg,
        "rsl_rl_cfg_entry_point": Go2BallKickPPORunnerCfg,
    },
)
