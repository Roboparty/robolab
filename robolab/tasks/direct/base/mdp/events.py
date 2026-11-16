# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math

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


def randomize_foot_torsional_friction(env, env_ids, friction_range=(0.005, 0.2)):
    if getattr(env.scene["robot"].data, "__backend_name__", None) != "newton":
        return

    import warp as wp
    from newton import ModelFlags

    solver = env.sim.physics_manager._solver
    geom_names = [solver.mj_model.geom(i).name or "" for i in range(solver.mj_model.ngeom)]
    foot_geoms = [i for i, name in enumerate(geom_names) if "/left_foot_" in name or "/right_foot_" in name]
    if not foot_geoms:
        raise RuntimeError("No RP1 foot collision geometries found for torsional friction randomization")

    mapping = wp.to_torch(solver.mjc_geom_to_newton_shape).long()
    friction = wp.to_torch(solver.model.shape_material_mu_torsional)
    geom_body = wp.to_torch(solver.mjw_model.geom_bodyid)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=friction.device)
    shapes = mapping[env_ids][:, foot_geoms]
    if (shapes < 0).any():
        raise RuntimeError("RP1 foot collision geometries are missing Newton shapes")
    values = torch.empty(len(env_ids), device=friction.device).uniform_(
        math.log(friction_range[0]), math.log(friction_range[1])
    ).exp_()
    friction[shapes] = values[:, None]
    # Equal-priority contacts use the larger coefficient from foot and terrain.
    ground_shapes = mapping[:, geom_body == 0].flatten()
    friction[ground_shapes[ground_shapes >= 0]] = friction_range[0]
    solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
