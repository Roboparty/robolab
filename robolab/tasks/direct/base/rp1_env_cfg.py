# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RP1 24-DOF variants of the Direct locomotion tasks."""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG as ISAACLAB_ROUGH_TERRAINS_CFG
from isaaclab_physx.physics import PhysxCfg

from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_BODY_ORDER

from isaaclab.managers import RewardTermCfg as RewTerm
from .base_config import BaseEnvCfg, RewardCfg
from . import mdp
from .scene_cfg import SceneCfg, configure_self_contact_sensors
from robolab.utils.physics import PhysicsPresetCfg, RoughPhysicsPresetCfg


@configclass
class RP1SceneCfg(SceneCfg):
    """RP1 contact filters and scanners for Direct locomotion tasks."""

    def __init__(self, config, physics_dt, step_dt):
        super().__init__(config=config, physics_dt=physics_dt, step_dt=step_dt)
        configure_self_contact_sensors(self, "contact_sensor", physics_dt, RP1_BODY_ORDER)
        if config.height_scanner.enable_height_scan:
            base_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
            for side in ("left", "right"):
                scanner = getattr(self, f"{side}_feet_scanner")
                scanner.prim_path = (
                    f"{base_path}/{side}_hip_pitch_link/{side}_hip_roll_link/{side}_hip_yaw_link/"
                    f"{side}_knee_link/{side}_ankle_pitch_link/{side}_ankle_roll_link"
                )
                scanner.debug_vis = False
            self.height_scanner.prim_path = base_path
        else:
            self.left_feet_scanner = None
            self.right_feet_scanner = None


@configclass
class RP1RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2, params={})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={})
    energy = RewTerm(func=mdp.energy, weight=-0.0001, params={})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-05, params={})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-07, params={})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2, params={"zero_on_reset": True})
    action_smoothness_l2 = RewTerm(func=mdp.action_jerk_l2, weight=-0.02)
    joint_rated_velocity = RewTerm(func=mdp.joint_rated_velocity_limit, weight=-0.05)
    joint_rated_effort = RewTerm(func=mdp.joint_rated_effort_limit, weight=-0.04)
    angular_momentum_rate_l2 = RewTerm(func=mdp.angular_momentum_rate_l2, weight=-5e-5)
    yaw_angular_momentum_l2 = RewTerm(func=mdp.yaw_angular_momentum_l2, weight=-1e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_self_contacts,
        weight=-0.5,
        params={
            "ground_sensor_names": tuple("self_contact_" + body for body in RP1_BODY_ORDER),
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor"
            )
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0, params={})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "velocity_reference": "link",
            "contact_history_length": 3,
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.003,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "min": 0.1,
            "max": 0.5,
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.1, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={})
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                    ".*_ankle.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*"])},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
        },
    )
    upward = RewTerm(func=mdp.upward, weight=0.4, params={})
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.2,
        params={
            "pos_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={
            "replace_invalid_hits": True,
            "force_type": "total",
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "ankle_height": 0.04,
            "threshold": 0.02,
        },
    )


@configclass
class RP1FlatEnvCfg(BaseEnvCfg):
    reward = RP1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.noise.noise_scales.joint_vel = 0.5
        self.action_space = 24
        self.observation_space = 243
        self.state_space = 432
        self.robot.actor_obs_history_length = 3
        self.robot.critic_obs_history_length = 3
        self.robot.observation_history_layout = "term_major"
        self.robot.preserve_feet_order = True
        self.robot.contact_force_type = "total"
        self.robot.replace_invalid_height_hits = True
        self.normalization.height_scan_invalid_value = 0.0
        self.noise.add_height_scan_noise = False
        self.normalization.clip_observations = float("inf")
        self.normalization.clip_actions = float("inf")
        self.scene_context.terrain_type = "plane"
        self.scene_context.robot = RP1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.sim.physics = PhysicsPresetCfg()
        self.sim.physics.isaacsim_physx = PhysxCfg()
        self.robot.terminate_contacts_body_names = [r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"]
        self.robot.feet_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.randomize_rigid_body_com.params["com_range"] = {axis: (-0.04, 0.04) for axis in ("x", "y", "z")}
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material.params["make_consistent"] = False
        self.events.randomize_joint_default_pos = EventTerm(
            func=mdp.randomize_default_joint_pos, mode="startup",
        )
        self.events.push_robot.interval_range_s = (1.0, 3.0)
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.scale_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["friction_distribution_params"] = (0.8, 1.2)
        self.scene = RP1SceneCfg(config=self.scene_context, physics_dt=self.sim.dt, step_dt=self.decimation * self.sim.dt)


@configclass
class RP1RoughEnvCfg(RP1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.state_space = 993
        self.scene_context.num_envs = 8192
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = ISAACLAB_ROUGH_TERRAINS_CFG.copy()
        self.scene_context.terrain_generator.curriculum = True
        self.scene_context.terrain_generator.horizontal_scale = 0.1
        self.sim.physics = RoughPhysicsPresetCfg()
        self.scene = RP1SceneCfg(config=self.scene_context, physics_dt=self.sim.dt, step_dt=self.decimation * self.sim.dt)
        self.reward.feet_height.params["sensor_cfg1"] = SceneEntityCfg("left_feet_scanner")
        self.reward.feet_height.params["sensor_cfg2"] = SceneEntityCfg("right_feet_scanner")
