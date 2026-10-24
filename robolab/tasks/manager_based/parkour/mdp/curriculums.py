from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())



def modify_rewards_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    init_weight: float,
    final_weight: float,
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
    step_size: float = 0.02,
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to gradually adjust a reward term's weight from ``init_weight`` toward
    ``final_weight`` when the robot tracks its commanded velocity well (high score), and step
    it back toward ``init_weight`` when the robot tracks poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        term_name: Name of the reward term whose weight will be modified.
        init_weight: Initial (easy) weight of the reward term.
        final_weight: Final (strict) weight to ramp toward when tracking is good.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear
            velocity tracking score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the weight is moved
            back toward ``init_weight``.
            If the score is above the upper threshold (good tracking), the weight is moved
            toward ``final_weight``.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular
            velocity tracking score (exponential kernel). Similar logic as ``lin_vel_threshold``.
        step_size: Fractional step taken toward the target weight each time this curriculum
            is triggered (0.02 = move 2% of the remaining gap).
        group_name: Name of the reward group that contains ``term_name``. If ``None``, the
            first group is used (matches ``MultiRewardManager`` behavior).

    Returns:
        The global mean reward term weight as a tensor scalar, for logging.
    """
    # extract the used quantities
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    # decide whether to ramp up (toward final_weight) or back down (toward init_weight)
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up

    # update per-environment weights for the specified envs only
    per_env_weights = env.reward_manager.get_per_env_term_weights(term_name)
    # normalize env_ids to tensor for indexing
    if isinstance(env_ids, slice):
        env_idx = torch.arange(env.num_envs, device=per_env_weights.device)[env_ids]
    elif not isinstance(env_ids, torch.Tensor):
        env_idx = torch.tensor(list(env_ids), dtype=torch.long, device=per_env_weights.device)
    else:
        env_idx = env_ids.to(device=per_env_weights.device)
    current = per_env_weights[env_idx]
    # move_up and move_down are boolean tensors aligned with env_ids
    move_up = move_up.to(dtype=current.dtype, device=current.device)
    move_down = move_down.to(dtype=current.dtype, device=current.device)
    # per-env updates
    current = current + (final_weight - current) * step_size * move_up
    current = current + (init_weight - current) * step_size * move_down
    # write back only for these envs
    env.reward_manager.set_term_weight_for_envs(term_name, env_idx, current)

    # Log the global mean so the curve reflects the full population, not only the reset batch.
    global_weights = env.reward_manager.get_per_env_term_weights(term_name)
    return global_weights.mean()

