# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RP1 24-DoF BeyondMimic task configuration."""

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robolab import ROBOLAB_ROOT_DIR
from robolab.tasks.direct.base import mdp as locomotion_mdp
from robolab.tasks.direct.base.scene_cfg import configure_self_contact_sensors
from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_JOINT_ORDER, RP1_BODY_ORDER
from robolab.tasks.manager_based.beyondmimic import mdp

from .beyondmimic_env_cfg import BeyondMimicEnvCfg, ObservationsCfg, RewardsCfg


RP1_TRACKING_BODY_NAMES = (
    "base_link", "waist_yaw_link", "left_hip_roll_link", "right_hip_roll_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_knee_link", "right_knee_link",
    "left_elbow_link", "right_elbow_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_link", "right_wrist_link",
)


@configclass
class RP1BeyondMimicObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        motion_preview = ObsTerm(func=mdp.motion_preview, params={"command_name": "motion"})
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class StudentCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    student: StudentCfg = StudentCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RP1BeyondMimicRewardsCfg(RewardsCfg):
    action_jerk_l2 = RewTerm(func=locomotion_mdp.action_jerk_l2, weight=-0.02)
    angular_momentum_rate_l2 = RewTerm(func=locomotion_mdp.angular_momentum_rate_l2, weight=-5e-5)
    yaw_angular_momentum_l2 = RewTerm(func=locomotion_mdp.yaw_angular_momentum_l2, weight=-1e-3)
    joint_velocity_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_energy = RewTerm(
        func=locomotion_mdp.energy, weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_rated_velocity = RewTerm(
        func=locomotion_mdp.joint_rated_velocity_limit, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "all_joints"},
    )
    joint_rated_effort = RewTerm(
        func=locomotion_mdp.joint_rated_effort_limit, weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": "all_joints"},
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide, weight=-0.3,
        params={
            "velocity_reference": "link",
            "contact_history_length": 3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    contact_force = RewTerm(
        func=mdp.contact_forces, weight=-1e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold": 600.0,
        },
    )
    undesired_contacts = RewTerm(
        func=locomotion_mdp.undesired_self_contacts, weight=-0.5,
        params={
            "ground_sensor_names": tuple("self_contact_" + body for body in RP1_BODY_ORDER),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
            ),
        },
    )


@configclass
class RP1BeyondMimicEnvCfg(BeyondMimicEnvCfg):
    observations: RP1BeyondMimicObservationsCfg = RP1BeyondMimicObservationsCfg()
    rewards: RP1BeyondMimicRewardsCfg = RP1BeyondMimicRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.history_length = 5
        configure_self_contact_sensors(self.scene, "contact_forces", self.sim.dt, RP1_BODY_ORDER)
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.joint_acc_l2.weight = -5e-7
        self.actions.joint_pos.joint_names = list(RP1_JOINT_ORDER)
        self.actions.joint_pos.preserve_order = True
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rp1_bm", "dance1_subject2_smooth6hz.npz",
        )
        self.commands.motion.quaternion_order = "xyzw"
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = list(RP1_TRACKING_BODY_NAMES)
        self.commands.motion.adaptive_kernel_size = 2
        self.commands.motion.reset_on_motion_end = True
        self.episode_length_s = 10.0

        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "waist_yaw_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.base_com.params["asset_cfg"].body_names = "waist_yaw_link"
        self.events.base_com.params["com_range"] = {axis: (-0.04, 0.04) for axis in ("x", "y", "z")}
        self.events.scale_joint_parameters.params["friction_distribution_params"] = (0.8, 1.2)
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion", "threshold": 0.25,
                "body_names": ["left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link"],
            },
        )
