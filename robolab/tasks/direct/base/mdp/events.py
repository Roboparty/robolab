# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def randomize_default_joint_pos(env, env_ids, offset_range=(-0.01, 0.01)):
    robot = env.scene["robot"]
    default = _as_torch(robot.data.default_joint_pos)
    if not hasattr(robot.data, "default_joint_pos_nominal"):
        robot.data.default_joint_pos_nominal = default[0].clone()
    if env_ids is None:
        env_ids = torch.arange(default.shape[0], device=default.device)
    default[env_ids] += torch.empty_like(default[env_ids]).uniform_(*offset_range)
    if hasattr(env, "action_manager"):
        term = env.action_manager.get_term("joint_pos")
        term._offset[env_ids] = default[env_ids][:, term._joint_ids]
