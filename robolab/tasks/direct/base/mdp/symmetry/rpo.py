# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _joint_mirror(start=0):
    indices = [start + 1, start, start + 2]
    for index in range(start + 3, start + 23, 2):
        indices.extend((index + 1, index))
    signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    return indices, signs


@lru_cache(maxsize=None)
def _frame_mirror(critic, interrupt, scan_rows, scan_cols):
    indices = list(range(9))
    signs = [-1, 1, -1, 1, -1, 1, 1, -1, -1]
    for offset in (9, 32, 55):
        joint_indices, joint_signs = _joint_mirror(offset)
        indices.extend(joint_indices)
        signs.extend(joint_signs)
    if interrupt:
        indices.append(78)
        signs.append(1)
    width = len(indices)
    if critic:
        indices.extend(width + index for index in (0, 1, 2, 4, 3, 8, 9, 10, 5, 6, 7, 12, 11, 14, 13))
        signs.extend((1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1))
        for offset in (width + 15, width + 38):
            joint_indices, joint_signs = _joint_mirror(offset)
            indices.extend(joint_indices)
            signs.extend(joint_signs)
    width = len(indices)
    for row in range(scan_rows):
        indices.extend(width + (scan_rows - 1 - row) * scan_cols + col for col in range(scan_cols))
    signs.extend([1] * (scan_rows * scan_cols))
    return indices, signs


def mirror_observation(env, values, critic=False):
    env = getattr(env, "unwrapped", env)
    interrupt = getattr(getattr(env.cfg, "interrupt", None), "use_interrupt", False)
    rows = cols = 0
    scanner = env.cfg.scene_context.height_scanner
    if scanner.enable_height_scan and (critic or scanner.enable_height_scan_actor):
        pattern = env.cfg.scene.height_scanner.pattern_cfg
        cols = int(round(pattern.size[0] / pattern.resolution)) + 1
        rows = int(round(pattern.size[1] / pattern.resolution)) + 1
    indices, signs = _frame_mirror(critic, interrupt, rows, cols)
    history = env.cfg.robot.critic_obs_history_length if critic else env.cfg.robot.actor_obs_history_length
    if values.shape[-1] != len(indices) * history:
        raise ValueError("RPO observation width does not match its configured history.")
    frames = values.reshape(*values.shape[:-1], history, len(indices))
    return (frames[..., indices] * values.new_tensor(signs)).reshape(values.shape)


def mirror_actions(actions):
    indices, signs = _joint_mirror()
    return actions[..., indices] * actions.new_tensor(signs)


@torch.no_grad()
def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        mirrored = obs.clone()
        mirrored["policy"] = mirror_observation(env, obs["policy"])
        if "critic" in obs.keys():
            mirrored["critic"] = mirror_observation(env, obs["critic"], critic=True)
        obs_aug = torch.cat((obs, mirrored), dim=0)
    actions_aug = None if actions is None else torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug
