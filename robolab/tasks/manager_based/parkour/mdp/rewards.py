from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.assets import RigidObject, Articulation
import isaaclab.utils.math as math_utils
from robolab.sensors.volume_points.points_generator import grid3d_points_generator
from robolab.sensors.volume_points.points_generator_cfg import Grid3dPointsGeneratorCfg

if TYPE_CHECKING:
    from robolab.sensors.volume_points import VolumePoints
    from isaaclab.envs import ManagerBasedRLEnv


def _body_names(asset, cfg: SceneEntityCfg) -> list[str]:
    if isinstance(cfg.body_ids, slice):
        return asset.body_names[cfg.body_ids]
    return asset.find_bodies(cfg.body_names, preserve_order=cfg.preserve_order)[1]


def _matching_body_ids(asset, body_names: list[str]) -> list[int]:
    return asset.find_bodies([re.escape(name) for name in body_names], preserve_order=True)[0]


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


# def stand_still(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     threshold: float = 0.15,
#     offset: float = 1.0,
# ) -> torch.Tensor:
#     """Penalize moving when there is no velocity command."""
#     asset = env.scene[asset_cfg.name]
#     dof_error = torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
#     return (
#         (dof_error - offset)
#         * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
#         * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
#     )

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    joint_pos_weight: float = 0.6,
    joint_vel_weight: float = 0.05,
    body_vel_weight: float = 0.35,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene["robot"]
    cmd = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) + torch.abs(env.command_manager.get_command(command_name)[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b.torch[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = joint_pos_weight *torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
    vel_reward = joint_vel_weight * torch.sum(torch.abs(asset.data.joint_vel.torch), dim=1)
    body_vel_reward = body_vel_weight * body_vel
    reward = torch.where(
        cmd > 0.1,
        0.0,
        pos_reward + vel_reward + body_vel_reward,
    )
    return reward

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)



def rpo_thigh_yaw_joint_sign_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize right thigh yaw below zero and left thigh yaw above zero."""
    asset = env.scene[asset_cfg.name]
    right_thigh_yaw_joint_index = asset.joint_names.index("right_thigh_yaw_joint")
    left_thigh_yaw_joint_index = asset.joint_names.index("left_thigh_yaw_joint")
    left_thigh_yaw_penalty = torch.where(asset.data.joint_pos.torch[:, left_thigh_yaw_joint_index] > 0.0, asset.data.joint_pos.torch[:, left_thigh_yaw_joint_index], 0.0)
    right_thigh_yaw_penalty = torch.where(asset.data.joint_pos.torch[:, right_thigh_yaw_joint_index] < 0.0, -asset.data.joint_pos.torch[:, right_thigh_yaw_joint_index], 0.0)
    return left_thigh_yaw_penalty + right_thigh_yaw_penalty

def body_distance_y(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.torch.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    feet_xy = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :2]
    left_foot_xy, right_foot_xy = feet_xy[:, 0], feet_xy[:, 1]
    heading_w = asset.data.heading_w.torch

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize vertical foot velocity and acceleration during ground contact."""
    asset = env.scene["robot"]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _matching_body_ids(asset, _body_names(contact_sensor, sensor_cfg))
    foot_vel_z = asset.data.body_vel_w.torch[:, body_ids, 2]
    foot_acc_z = asset.data.body_acc_w.torch[:, body_ids, 2]
    contact_force_z = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(contact_force_z) > 0.0
    penalty = ((foot_vel_z.square() + foot_acc_z.square()) * in_contact.float()).sum(dim=1)
    return penalty

def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b.torch[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    left_quat = feet_quat[:, 0]
    left_projected_gravity = quat_apply_inverse(left_quat, gravity_w)
    right_quat = feet_quat[:, 1]
    right_projected_gravity = quat_apply_inverse(right_quat, gravity_w)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_ids = _matching_body_ids(contact_sensor, _body_names(asset, asset_cfg))
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contact_ids = _matching_body_ids(contact_sensor, _body_names(asset, asset_cfg))
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_ids], dim=-1), dim=1)[0] > 1
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w.torch[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w.torch[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    feet_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    left_height, right_height = feet_height[:, 0], feet_height[:, 1]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :][:, 0]
    gravity_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    link_projected_gravity = quat_apply_inverse(link_quat, gravity_w)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 2 * forces_z, dim=1).float()
    return reward

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 1.0) / 1.0
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 1.0) / 1.0
    return reward


def volume_points_penetration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)

    return torch.sum(velocity_times_penetration, dim=-1)


def volume_points_penetration_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    enable_terrain_foot_weights: bool = False,
    stairs_weight_min: float = 0.2,
    stairs_weight_max: float = 1.0,
    debug_print_terrain: bool = False,
) -> torch.Tensor:
    """Penalize volume-point penetration into virtual edge obstacles.

    With ``enable_terrain_foot_weights``, foot-local x maps heel (x_min) to toe (x_max).
    Up-stairs terrains (``pyramid_stairs_inv``) use toe-heavy weights; down-stairs
    (``pyramid_stairs``) use heel-heavy weights. Other terrains use mid-foot-heavy
    weights with heel and toe at ``stairs_weight_min`` and mid-foot at ``stairs_weight_max``.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    num_envs, num_bodies, num_points, _ = penetration.shape
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)

    if enable_terrain_foot_weights:
        terrain = env.scene.terrain
        if terrain.cfg.terrain_type == "generator":
            gen_cfg = terrain.cfg.terrain_generator
            sub_names = list(gen_cfg.sub_terrains.keys())
            terrain_gen = getattr(terrain, "terrain_generator", None)
            if terrain_gen is None or not hasattr(terrain_gen, "get_subterrain_indices"):
                raise RuntimeError(
                    "terrain_generator with subterrain_index_grid is required for enable_terrain_foot_weights. "
                    "Use robolab.terrains.TerrainImporter with terrain_type='generator' and FiledTerrainGeneratorCfg."
                )
            sub_idx_per_env = terrain_gen.get_subterrain_indices(
                terrain.terrain_levels, terrain.terrain_types, device=env.device
            )

            if debug_print_terrain:
                for i in range(num_envs):
                    name = sub_names[sub_idx_per_env[i].item()]
                    if "pyramid_stairs_inv" in name:
                        label = "up"
                    elif "pyramid_stairs" in name and "inv" not in name:
                        label = "down"
                    else:
                        label = "other"
                    print(f"env {i}: {label}, {name}")

            mask_up = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            mask_down = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            for sub_idx, name in enumerate(sub_names):
                if "pyramid_stairs_inv" in name:
                    mask_up |= sub_idx_per_env == sub_idx
                elif "pyramid_stairs" in name and "inv" not in name:
                    mask_down |= sub_idx_per_env == sub_idx

            points_cfg = volume_sensor.cfg.points_generator
            local_points = grid3d_points_generator(points_cfg).to(env.device)
            x_frac = (local_points[:, 0] - points_cfg.x_min) / (points_cfg.x_max - points_cfg.x_min + 1e-8)
            x_frac = x_frac.clamp(0.0, 1.0)
            weight_span = stairs_weight_max - stairs_weight_min
            w_toe_heavy = stairs_weight_min + weight_span * x_frac
            w_heel_heavy = stairs_weight_max - weight_span * x_frac
            w_mid_heavy = stairs_weight_min + weight_span * (1.0 - torch.abs(2.0 * x_frac - 1.0))

            env_w = w_mid_heavy.unsqueeze(0).repeat(num_envs, 1)
            if mask_up.any():
                env_w[mask_up] = w_toe_heavy.unsqueeze(0)
            if mask_down.any():
                env_w[mask_down] = w_heel_heavy.unsqueeze(0)

            velocity_times_penetration = velocity_times_penetration.view(num_envs, num_bodies, num_points)
            velocity_times_penetration = velocity_times_penetration * env_w.unsqueeze(1)
            velocity_times_penetration = velocity_times_penetration.flatten(1, 2)

    return torch.sum(velocity_times_penetration, dim=-1)


def step_safety(
    env: ManagerBasedRLEnv,
    volume_points_cfg: SceneEntityCfg,
    contact_forces_cfg: SceneEntityCfg,
    epsilon: float = 1e-5,
    once: bool = False,
) -> torch.Tensor:
    """A log based reward to encourage the robot to make contacts with no penetration to the virtual obstacles.
    Inspired by Deep Tracking Control and Robot Parkour Learning and Humanoid Parkour Learning.
    NOTE: make sure the contact forces sensor is selected for that the volume points sensors are interested in.
    aka. The number of selected bodies in the contact forces sensor should be the same as the number of selected bodies
    in all volume points sensors.
    NOTE: Be aware of the body order.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[volume_points_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_forces_cfg.name]
    contact_ids = _matching_body_ids(contact_sensor, volume_sensor.body_names)
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_, P_)
    penetration_depth_max = torch.max(penetration_depth, dim=-1)[0]  # (N, B_)
    if once:
        contacts = contact_sensor.compute_first_contact(env.step_dt).torch[:, contact_ids]  # (N, B_)
    else:
        contact_forces = contact_sensor.data.net_normal_forces_w_history.torch[:, :, contact_ids, :]  # (N, T, B_, 3)
        contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0  # (N, B_)

    rewards = -torch.log(penetration_depth_max + epsilon) * contacts
    return torch.sum(rewards, dim=-1)

def contact_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_penalty: bool = False,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize body sliding.

    This function penalizes the agent for sliding its body on the ground. The reward is computed as the
    norm of the linear velocity of the body multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the body are in contact with the ground.
    """
    # Penalize body sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact_ids = _matching_body_ids(contact_sensor, _body_names(asset, asset_cfg))
    contacts = (
        contact_sensor.data.net_normal_forces_w_history.torch[:, :, contact_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )

    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    body_ang_vel = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    if ang_vel_penalty:
        reward = reward + torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def motors_power_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    power_j = asset.data.applied_torque.torch * asset.data.joint_vel.torch  # (batch_size, num_joints)
    if normalize_by_stiffness:
        for _, actuator in asset.actuators.items():
            power_j[:, actuator.joint_indices] /= actuator.stiffness
    power_j = power_j[:, asset_cfg.joint_ids]  # (batch_size, num_selected_joints)
    power = torch.sum(torch.square(power_j), dim=-1)  # (batch_size,)
    if normalize_by_num_joints:
        power /= power_j.shape[-1]
    return power
    
def applied_torque_limits_by_ratio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
):
    """Penalize when the applied torque excceed certain ratio of the joint torque limit."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_effort_limits = asset.data.joint_effort_limits.torch  # (num_envs, num_joints)
    joint_effort_limits = joint_effort_limits[:, asset_cfg.joint_ids]

    applied_torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(applied_torque)

    out_of_limits = (applied_torque - joint_effort_limits * limit_ratio).clip(min=0)
    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)  # (num_envs,)

    return out_of_limits_err
