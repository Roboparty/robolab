from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.parkour.mdp.symmetry import rpo


@configclass
class RslRlAmpCfg:
    """Configuration for adversarial motion priors."""

    disc_obs_buffer_size: int = 1000
    grad_penalty_scale: float = 10.0
    disc_trunk_weight_decay: float = 1.0e-4
    disc_linear_weight_decay: float = 1.0e-2
    disc_learning_rate: float = 1.0e-5
    disc_max_grad_norm: float = 1.0

    @configclass
    class AMPDiscriminatorCfg:
        """Discriminator architecture and style reward settings."""

        hidden_dims: list[int] = MISSING
        activation: str = "elu"
        style_reward_scale: float = 1.0
        task_style_lerp: float = 0.0

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with an AMP discriminator."""

    class_name: str = "rsl_rl.algorithms.ppo_amp:PPOAMP"
    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()


@configclass
class RslRlEncoderMoEModelCfg(RslRlMLPModelCfg):
    class_name: str = "rsl_rl.models.encoder_model:EncoderMoEModel"
    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = []
    hidden_dims: list[int] = [256, 128, 64]
    obs_normalization: bool = False
    activation: str = "elu"
    encoder_obs_groups: list[str] = ["depth"]
    encoder_cfg: dict = {
        "channels": [16, 32, 64],
        "kernel_sizes": [3, 3, 3],
        "strides": [1, 2, 1],
        "hidden_sizes": [256],
        "output_size": 256,
        "paddings": [1, 1, 1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
        "input_dtype": "uint8",
        "input_range": (0.0, 255.0),
    }


@configclass
class RPOParkourAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    export_policy_func: str = "robolab.tasks.manager_based.parkour.agents.exporter:export_policy"
    num_steps_per_env = 24
    compact_depth_history = True
    depth_history_offsets = (35, 30, 25, 20, 15, 10, 5, 0)
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rpo_parkour"
    wandb_project = "rpo_parkour"
    obs_groups = {
        "actor": ["policy", "depth"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    actor = RslRlEncoderMoEModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log")
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="rsl_rl.algorithms.ppo_amp:PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=5.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="ReLU",
                style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            loss_type="LSGAN",
        ),
    )
