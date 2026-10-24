# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from robolab.tasks.direct.base import mdp as locomotion_mdp
from robolab.tasks.direct.base.scene_cfg import configure_self_contact_sensors

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_JOINT_ORDER, RP1_BODY_ORDER
from robolab.tasks.manager_based.amp import mdp
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg

AMP_NUM_STEPS = 3

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_link",
    "right_wrist_link",
]


@configclass
class RP1AmpRewards:
    """Reward terms for the MDP."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)

    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    action_jerk_l2 = RewTerm(func=locomotion_mdp.action_jerk_l2, weight=-0.02)
    joint_rated_velocity = RewTerm(func=locomotion_mdp.joint_rated_velocity_limit, weight=-0.05)
    joint_rated_effort = RewTerm(func=locomotion_mdp.joint_rated_effort_limit, weight=-0.04)
    angular_momentum_rate_l2 = RewTerm(func=locomotion_mdp.angular_momentum_rate_l2, weight=-5e-5)
    yaw_angular_momentum_l2 = RewTerm(func=locomotion_mdp.yaw_angular_momentum_l2, weight=-1e-3)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_ankle_roll.*"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle_pitch.*"])},
    )

    # -- Feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": 0.4,
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "ankle_height": 0.04,
            "threshold": 0.02,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll.*"]), "min": 0.1, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.feet_distance_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.1, "max": 0.35},
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=0,
        params={
            "velocity_reference": "link",
            "contact_history_length": 3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=locomotion_mdp.undesired_self_contacts,
        weight=-0.5,
        params={
            "ground_sensor_names": tuple("self_contact_" + body for body in RP1_BODY_ORDER),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class RP1AmpEnvCfg(AmpEnvCfg):
    rewards: RP1AmpRewards = RP1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RP1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        configure_self_contact_sensors(self.scene, "contact_forces", self.sim.dt, RP1_BODY_ORDER)

        self.actions.joint_pos.joint_names = list(RP1_JOINT_ORDER)
        self.actions.joint_pos.preserve_order = True
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.quaternion_order = "wxyz"
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
        }
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(ROBOLAB_ROOT_DIR, "data", "motions", "rp1_amp")
        # Balanced integer weights with equal forward/mirror and left/right sampling.
        # See the data README for durations and normalized probabilities.
        self.motion_data.motion_dataset.motion_data_weights = {
            "neutral_walk_ff_180_R_002__A534": 3,
            "neutral_walk_ff_180_R_002__A534_M": 3,
            "move_l": 2,
            "move_r": 2,
            "stand_still": 1,
            "turn_l": 2,
            "turn_r": 2,
            "move_back": 3,
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.5, n_max=0.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.randomize_rigid_body_com.params["com_range"] = {axis: (-0.04, 0.04) for axis in ("x", "y", "z")}
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material.params["make_consistent"] = False
        self.events.scale_joint_parameters.params["friction_distribution_params"] = (0.8, 1.2)
        self.events.randomize_joint_default_pos = EventTerm(
            func=locomotion_mdp.randomize_default_joint_pos, mode="startup",
        )
        self.events.push_robot.interval_range_s = (1.0, 3.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_yaw_link"]

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.25

        # alive
        self.rewards.alive.weight = 0.15

        # base
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0

        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -5e-7
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_deviation_hip.weight = -0.03
        self.rewards.joint_deviation_waist.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -0.01
        self.rewards.joint_deviation_arms.weight = -0.06

        # feet
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_height.weight = 0.2
        self.rewards.knee_distance.weight = 0.1
        self.rewards.feet_distance.weight = 0.1
        self.rewards.feet_slide.weight = -0.3

        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
        )

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_hip_pitch_link",
            ".*_hip_roll_link",
            "waist_yaw_link",
            ".*_shoulder_pitch_link",
            ".*_elbow_link",
            ".*_wrist.*",
        ]
        if self.__class__.__name__ == "RP1AmpEnvCfg":
            self.disable_zero_weight_rewards()

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)


@configclass
class RP1AmpEnvCfg_PLAY(RP1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
