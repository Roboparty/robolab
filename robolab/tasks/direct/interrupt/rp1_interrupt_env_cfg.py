# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import preset
from isaaclab_physx.physics import PhysxCfg

from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_BODY_ORDER
from robolab.tasks.direct.base import mdp
from robolab.tasks.direct.base.base_config import BaseEnvCfg, RewardCfg
from robolab.tasks.direct.base.scene_cfg import SceneCfg, configure_self_contact_sensors
from robolab.utils.physics import PhysicsPresetCfg

from .interrupt_env_cfg import ArmClearanceCfg, InterruptCfg

RP1_INTERRUPT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
RP1_INTERRUPT_LOWER_BOUND = (-3.14, -0.25, -1.57, -1.57, -3.14, -3.14, -1.57, -1.2)
RP1_INTERRUPT_SCALE = (6.28, 3.39, 3.14, 2.77, 6.28, 3.39, 3.14, 2.77)


RP1_INTERRUPT_GROUND_CONTACT_PATHS = (
    "waist_roll_link",
    "waist_roll_link/waist_yaw_link",
    "left_hip_pitch_link/left_hip_roll_link",
    "left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link",
    "right_hip_pitch_link/right_hip_roll_link",
    "right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link",
)


@configclass
class RP1InterruptRewardCfg(RewardCfg):
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*", ".*_wrist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"]
            )
        },
    )
    joint_deviation_interrupt = RewTerm(
        func=mdp.joint_deviation_interrupt,
        weight=-1.0,
        params={
            "asset_cfg1": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*"]
            ),
            "asset_cfg2": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch.*"]),
            "weight1": 1.0,
            "weight2": 0.06,
        },
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
        func=mdp.stand_still_interrupt,
        weight=-0.2,
        params={
            "pos_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "interrupt_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder.*", ".*_elbow.*"]),
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
    action_penalty = RewTerm(
        func=mdp.action_penalty_interrupt,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder.*", ".*_elbow.*"])},
    )


@configclass
class RP1InterruptEnvCfg(BaseEnvCfg):
    reward = RP1InterruptRewardCfg()

    interrupt = InterruptCfg(
        arm_joint_signs=(1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0),
        arm_clearance=ArmClearanceCfg(
            roll_clearance=0.2,
            backward_pitch_limit=0.2,
            pitch_per_roll=2.0,
            forward_yaw_margin=0.4,
            outward_yaw_per_roll=2.0,
            low_roll_elbow_flexion=1.2,
            elbow_flexion_per_roll=0.5,
            reverse_elbow_roll_start=1.0,
            reverse_elbow_range=0.5,
            reverse_elbow_roll_span=0.57,
            overhead_roll_start=2.5,
            overhead_yaw_slope=1.57 / 0.93,
            overhead_yaw_margin=0.3,
            reverse_elbow_yaw_scale=0.8,
        ),
        use_interrupt=True,
        # Preserve RPO's 4 m threshold on RP1's flat scene.
        curriculum_pass_distance=4.0,
        interrupt_joint_names=list(RP1_INTERRUPT_JOINT_NAMES),
        interrupt_scale=list(RP1_INTERRUPT_SCALE),
        interrupt_lower_bound=list(RP1_INTERRUPT_LOWER_BOUND),
    )
    interrupt_vis = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )

    def __post_init__(self):
        super().__post_init__()
        self.noise.noise_scales.joint_vel = 0.5
        self.action_space = 24
        self.observation_space = 246
        self.state_space = 435
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
        self.scene = SceneCfg(config=self.scene_context, physics_dt=self.sim.dt, step_dt=self.decimation * self.sim.dt)
        configure_self_contact_sensors(self.scene, "contact_sensor", self.sim.dt, RP1_BODY_ORDER)
        if self.scene_context.height_scanner.enable_height_scan:
            base_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
            for side in ("left", "right"):
                scanner = getattr(self.scene, f"{side}_feet_scanner")
                scanner.prim_path = (
                    f"{base_path}/{side}_hip_pitch_link/{side}_hip_roll_link/{side}_hip_yaw_link/"
                    f"{side}_knee_link/{side}_ankle_pitch_link/{side}_ankle_roll_link"
                )
                scanner.debug_vis = False
            self.scene.height_scanner.prim_path = base_path
        else:
            self.scene.left_feet_scanner = None
            self.scene.right_feet_scanner = None
        # Match RPO Interrupt's torso and hip yaw/roll contact failures. RP1
        # splits the torso into two waist links. Filter to ground contacts so
        # incidental arm/hip self contact is penalized without ending a trial.
        self.robot.terminate_contacts_body_names = [
            "waist_.*_link",
            ".*_hip_yaw_link",
            ".*_hip_roll_link",
        ]
        self.robot.ground_contact_sensor_names = tuple(f"interrupt_ground_contact_{index}" for index in range(6))
        for name, path in zip(self.robot.ground_contact_sensor_names, RP1_INTERRUPT_GROUND_CONTACT_PATHS):
            setattr(
                self.scene,
                name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/" + path,
                    filter_prim_paths_expr=preset(
                        default=[],
                        isaacsim_physx=["/World/ground/terrain/GroundPlane/CollisionPlane"],
                        newton_mjwarp=[],
                    ),
                    filter_shape_prim_expr=preset(
                        default=["/World/ground/.*"],
                        isaacsim_physx=[],
                        newton_mjwarp=["/World/ground/.*"],
                    ),
                    history_length=self.decimation,
                    update_period=self.sim.dt,
                ),
            )
