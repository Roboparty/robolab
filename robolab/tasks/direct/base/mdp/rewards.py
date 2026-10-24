# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from .observations import feet_clearance
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from ..base_env import BaseEnv


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _contact_forces(sensor, force_type="normal", history=False):
    if force_type not in ("normal", "total"):
        raise ValueError(f"Unsupported contact force type: {force_type!r}")
    name = "net_normal_forces_w" if force_type == "normal" else "net_forces_w"
    return _as_torch(getattr(sensor.data, name + ("_history" if history else "")))


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w.torch[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b.torch[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids] * asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=-1)
    return reward


def action_rate_l2(env: BaseEnv, zero_on_reset: bool = False) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, :] - (
                env.previous_action if zero_on_reset else env.action_buffer.buffer[:, -2, :]
            )
        ),
        dim=1,
    )

def action_smoothness_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, :] - 2*env.action_buffer.buffer[:, -2, :] + env.action_buffer.buffer[:, -1, :] 
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_terminated


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg, force_type: str = "normal") -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    is_contact = _contact_forces(contact_sensor, force_type, history=True)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_mode_time = torch.where(is_contact, contact_time, air_time)
    single_stance = torch.sum(is_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_reference: str = "com", contact_history_length: int | None = None
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = contact_sensor.data.net_normal_forces_w_history.torch[:, :contact_history_length, sensor_cfg.body_ids]
    contacts = history.norm(dim=-1).amax(dim=1) > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    if velocity_reference not in ("com", "link"):
        raise ValueError(f"Unsupported foot velocity reference: {velocity_reference!r}")
    velocity = asset.data.body_link_lin_vel_w if velocity_reference == "link" else asset.data.body_lin_vel_w
    body_vel = velocity.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400,
    force_type: str = "normal"
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = torch.sum(torch.linalg.norm(_contact_forces(contact_sensor, force_type)[:, sensor_cfg.body_ids, :], dim=2), dim=1)
    reward = (reward - threshold).clamp(min=0.0, max=max_reward)
    return reward


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    body_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w.torch[:, body_id, :], gravity_w
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    return torch.sum(torch.sum(torch.square(body_orientation[:, :2, :]), dim=1), dim=-1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg, force_type: str = "normal") -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(_contact_forces(contact_sensor, force_type)[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(_contact_forces(contact_sensor, force_type)[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def body_distance_y(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.torch.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_without_cmd(env: BaseEnv, sensor_cfg: SceneEntityCfg, force_type: str = "normal") -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = _contact_forces(contact_sensor, force_type, history=True)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    reward = (torch.sum(contacts, dim=-1) == 2).float()
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward

def undesired_foothold(env: BaseEnv, sensor_cfg: SceneEntityCfg, sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    undesired_contacts = torch.stack(
        [
            torch.sum(
                (env.scene[sensor.name].data.pos_w.torch[:, 2].unsqueeze(1)
                - env.scene[sensor.name].data.ray_hits_w.torch[..., 2]
                - ankle_height) > 0.01,
                dim=-1
            ) / float(env.scene[sensor.name].data.ray_hits_w.torch.shape[1])
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    reward = torch.where(contacts, undesired_contacts, 0.0)
    return reward.sum(dim=1)

def upward(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = -asset.data.projected_gravity_b.torch[:, 2]
    return reward


def stand_still(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b.torch[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = pos_weight * torch.sum(torch.abs
        (asset.data.joint_pos.torch[:, pos_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, pos_cfg.joint_ids]), dim=1
    )
    vel_reward = vel_weight * torch.sum(torch.abs(asset.data.joint_vel.torch[:, vel_cfg.joint_ids]), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035, threshold: float = 0.05,
    force_type: str = "normal", replace_invalid_hits: bool = False):
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = _contact_forces(contact_sensor, force_type, history=True)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    sensors = tuple(cfg for cfg in (sensor_cfg1, sensor_cfg2) if cfg is not None)
    feet_height = feet_clearance(
        env, asset_cfg, sensors, ankle_height, replace_invalid_hits
    )
    # Compute single_stance mask
    single_stance = contacts.sum(dim=1) == 1
    # feet height should be closed to target feet height at the peak
    rew_pos = feet_height > threshold
    reward = torch.where(torch.logical_and(~contacts, single_stance.unsqueeze(-1)), rew_pos.float(), 0.0).sum(dim=1)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward

def joint_deviation_interrupt(env: BaseEnv, asset_cfg1: SceneEntityCfg, asset_cfg2: SceneEntityCfg, weight1: float, weight2: float) -> torch.Tensor:
    """Penalize joint deviation during interruption."""
    # extract the used quantities (to enable type-hinting)
    asset1: Articulation = env.scene[asset_cfg1.name]
    asset2: Articulation = env.scene[asset_cfg2.name]
    angle1 = asset1.data.joint_pos.torch[:, asset_cfg1.joint_ids] - asset1.data.default_joint_pos.torch[:, asset_cfg1.joint_ids]
    angle2 = asset2.data.joint_pos.torch[:, asset_cfg2.joint_ids] - asset2.data.default_joint_pos.torch[:, asset_cfg2.joint_ids]
    reward = weight1 * torch.sum(torch.abs(angle1), dim=1) + weight2 * torch.sum(torch.abs(angle2), dim=1)
    reward *= ~env.interrupt_mask
    return reward

def stand_still_interrupt(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    interrupt_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b.torch[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_joint_ids = list(set(pos_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    vel_joint_ids = list(set(vel_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    pos_reward = torch.where(env.interrupt_mask, 
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos.torch[:, pos_joint_ids] - asset.data.default_joint_pos.torch[:, pos_joint_ids]), dim=1),
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos.torch[:, pos_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, pos_cfg.joint_ids]), dim=1))
    vel_reward = torch.where(env.interrupt_mask, 
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel.torch[:, vel_joint_ids]), dim=1),
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel.torch[:, vel_cfg.joint_ids]), dim=1))
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward

def action_penalty_interrupt(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action magnitude during interruption."""
    reward = torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    reward *= env.interrupt_mask
    return reward


def _self_contact_count(normal, ground):
    normal = _as_torch(normal)[:, :3]
    if ground is None:
        raise RuntimeError("Self-contact reward requires terrain-filtered normal-force history.")
    ground = _as_torch(ground)[:, :3]
    if ground.shape[3] == 0:
        raise RuntimeError("Self-contact reward terrain filter matched no objects.")
    residual = normal - ground.sum(dim=3)
    return (residual.norm(dim=-1) > 1.0).any(dim=1).float().sum(dim=1)


def undesired_self_contacts(env, sensor_cfg, ground_sensor_names=()):
    data = env.scene.sensors[sensor_cfg.name].data
    if data.normal_force_matrix_w_history is not None:
        return _self_contact_count(data.net_normal_forces_w_history, data.normal_force_matrix_w_history)
    bodies = [
        env.scene.sensors[name].data
        for name in ground_sensor_names
    ]
    if not bodies or any(body.normal_force_matrix_w_history is None for body in bodies):
        raise RuntimeError("Self-contact reward requires a ground filter for every PhysX body.")
    # Batch the reward calculation instead of launching it separately for each body.
    return _self_contact_count(
        torch.cat([_as_torch(body.net_normal_forces_w_history) for body in bodies], dim=2),
        torch.cat([_as_torch(body.normal_force_matrix_w_history) for body in bodies], dim=2),
    )


def joint_rated_velocity_limit(env, asset_cfg=SceneEntityCfg("robot"), actuator_name="all_joints"):
    robot = env.scene[asset_cfg.name]
    actuator = robot.actuators[actuator_name]
    velocity = _as_torch(robot.data.joint_vel)[:, actuator.joint_indices]
    return torch.relu(velocity.abs() / actuator.rated_velocity - 1.0).square().sum(dim=1)


def joint_rated_effort_limit(env, asset_cfg=SceneEntityCfg("robot"), actuator_name="all_joints"):
    actuator = env.scene[asset_cfg.name].actuators[actuator_name]
    return torch.relu(_as_torch(actuator.applied_effort).abs() / actuator.rated_effort - 1.0).square().sum(dim=1)


def _angular_momentum(env):
    data = env.scene["robot"].data
    pos = _as_torch(data.body_com_pos_w)
    vel = _as_torch(data.body_com_lin_vel_w)
    mass = _as_torch(data.body_mass).unsqueeze(-1)
    total = mass.sum(dim=1).clamp(min=1e-6)
    com = (pos * mass).sum(dim=1) / total
    com_vel = (vel * mass).sum(dim=1) / total
    orbital = torch.cross(pos - com[:, None], vel - com_vel[:, None], dim=-1) * mass
    rotation = matrix_from_quat(_as_torch(data.body_com_quat_w))
    inertia = _as_torch(data.body_inertia).reshape(*pos.shape[:2], 3, 3)
    world_inertia = rotation @ inertia @ rotation.transpose(-1, -2)
    spin = (world_inertia @ _as_torch(data.body_com_ang_vel_w).unsqueeze(-1)).squeeze(-1)
    return (orbital + spin).sum(dim=1)


def _bounded(value, maximum):
    return torch.nan_to_num(value, nan=maximum, posinf=maximum, neginf=0.0).clamp(max=maximum)


def yaw_angular_momentum_l2(env):
    return _bounded(_angular_momentum(env)[:, 2].square(), 1e3)


class angular_momentum_rate_l2(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros(env.num_envs, 3, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        current = _angular_momentum(env)
        finite = current.isfinite().all(dim=-1)
        penalty = _bounded(((current - self.previous) / env.step_dt).square().sum(dim=-1), 1e4)
        result = torch.where(self.valid & finite, penalty, 0.0)
        self.previous.copy_(torch.where(finite[:, None], current, self.previous))
        self.valid.copy_(finite)
        return result


class action_jerk_l2(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = None
        self.previous_previous = None
        self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        self.steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env):
        action = env.action_manager.action if hasattr(env, "action_manager") else env.action_buffer.buffer[:, -1]
        if self.previous is None:
            self.previous = action.clone()
            self.previous_previous = action.clone()
        penalty = (action - 2.0 * self.previous + self.previous_previous).square().sum(dim=-1)
        result = torch.where(self.steps >= 2, penalty, 0.0)
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(action)
        self.steps += 1
        return result
