from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
import isaaclab.utils.math as math_utils
from robolab.sensors.volume_points import VolumePoints
from robolab.sensors.volume_points.points_generator import grid3d_points_generator
from robolab.sensors.volume_points.points_generator_cfg import Grid3dPointsGeneratorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 2])


def action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize action changes."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


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


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize squared joint deviation from default, optionally scaling the offset first."""
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(scale * angle), dim=1)

def link_close_y_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when two links are too close together in the heading-frame y distance."""
    asset = env.scene[asset_cfg.name]
    left_xy = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0], :2]
    right_xy = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w.torch
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)
    left_y = -sin_heading * left_xy[:, 0] + cos_heading * left_xy[:, 1]
    right_y = -sin_heading * right_xy[:, 0] + cos_heading * right_xy[:, 1]
    distance_y = torch.abs(left_y - right_y)
    return 1 - torch.exp(-torch.clamp(threshold - distance_y, min=0.0) / std**2)


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize vertical foot velocity and acceleration during ground contact."""
    asset = env.scene["robot"]
    body_ids = sensor_cfg.body_ids
    foot_vel_z = asset.data.body_vel_w.torch[:, body_ids, 2]
    foot_acc_z = asset.data.body_acc_w.torch[:, body_ids, 2]
    contact_force_z = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w.torch[:, body_ids, 2]
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
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b.torch[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())

def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
) -> torch.Tensor:
    """Penalize hanging sole while feet are in contact (hard clearance sum)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    is_contact = (
        torch.norm(
            contact_sensor.data.net_normal_forces_w.torch[:, contact_sensor_cfg.body_ids],
            dim=-1,
        )
        > 1
    )
    left_sensor = env.scene[left_height_scanner_cfg.name]
    right_sensor = env.scene[right_height_scanner_cfg.name]
    left_ray = left_sensor.data.ray_hits_w.torch[..., 2]
    right_ray = right_sensor.data.ray_hits_w.torch[..., 2]
    left_heights = torch.where(torch.isinf(left_ray), 0.0, left_ray)
    right_heights = torch.where(torch.isinf(right_ray), 0.0, right_ray)
    left_foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0], 2]
    right_foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[1], 2]
    left_clear = (
        torch.clamp(left_foot_z.unsqueeze(-1) - left_heights - height_offset, min=0.0, max=0.3)
        * is_contact[:, 0:1].float()
    )
    right_clear = (
        torch.clamp(right_foot_z.unsqueeze(-1) - right_heights - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2].float()
    )
    return torch.sum(left_clear, dim=-1) + torch.sum(right_clear, dim=-1)

def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0], :]
    gravity_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    link_projected_gravity = quat_apply_inverse(link_quat, gravity_w)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)

def link_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy angular velocity of specified link(s) in world frame using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    link_ang_vel_xy = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(link_ang_vel_xy), dim=(1, 2))

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
    return torch.exp(-ang_vel_error / std**2)


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
) -> torch.Tensor:
    """Penalize volume-point penetration into virtual edge obstacles.

    With ``enable_terrain_foot_weights``, sole points use a unified mid-heavy
    profile (heel/toe at ``stairs_weight_min``, mid-foot at ``stairs_weight_max``).
    """
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3)
    num_envs, num_bodies, num_points, _ = penetration.shape
    penetration = penetration.flatten(1, 2)
    penetration_depth = torch.norm(penetration, dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    points_vel = volume_sensor.data.points_vel_w.flatten(1, 2)
    points_vel_norm = torch.norm(points_vel, dim=-1)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth

    if enable_terrain_foot_weights:
        local_x = grid3d_points_generator(volume_sensor.cfg.points_generator).to(env.device)[:, 0]
        x_min, x_max = local_x.min(), local_x.max()
        x_frac = ((local_x - x_min) / (x_max - x_min + 1e-8)).clamp(0.0, 1.0)
        w = stairs_weight_min + (stairs_weight_max - stairs_weight_min) * (
            1.0 - 2.0 * torch.abs(x_frac - 0.5)
        )
        velocity_times_penetration = (
            velocity_times_penetration.view(num_envs, num_bodies, num_points)
            * w.view(1, 1, -1)
        ).flatten(1, 2)

    return torch.sum(velocity_times_penetration, dim=-1)


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize world-frame foot XY linear speed + yaw angular speed while in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    contacts = (
        contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :].norm(dim=-1) > threshold
    )
    v_lin_xy = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    omega_yaw = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, 2]
    penalty = v_lin_xy.norm(dim=-1) + omega_yaw.abs()
    return torch.sum(penalty * contacts, dim=1)

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

def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = forces.norm(dim=-1).amax(dim=1) > 1.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    # no reward for zero command
    command = env.command_manager.get_command(command_name)
    moving = command[:, :2].norm(dim=1) + command[:, 2].abs() > 0.01
    upright = (-asset.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 0.7) / 0.7
    return reward * moving * upright
