from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import RigidObject, Articulation


def push_by_setting_velocity_without_stand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges. No pushing when standing still."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w.torch[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    add_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    lin_vel = torch.norm(env.command_manager.get_command(command_name)[env_ids, :2], dim=1) > 0.15
    ang_vel = torch.abs(env.command_manager.get_command(command_name)[env_ids, 2]) > 0.15
    should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)

    vel_w += add_vel * should_push
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim_index(root_velocity=vel_w, env_ids=env_ids)
