# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh as lab_height_field_to_mesh

if TYPE_CHECKING:
    from isaaclab.terrains.height_field import HfTerrainBaseCfg


def height_field_to_mesh(func: Callable) -> Callable:
    """Supply GA's explicit origin while preserving RoboLab's center spawn height."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        terrain_size = cfg.size

        @functools.wraps(func)
        def with_origin(difficulty, cfg):
            heights = func(difficulty, cfg)
            border = int(cfg.border_width / cfg.horizontal_scale) + 1
            padded = np.pad(heights, border)
            x1, x2 = [int((terrain_size[0] * 0.5 + offset) / cfg.horizontal_scale) for offset in (-1, 1)]
            y1, y2 = [int((terrain_size[1] * 0.5 + offset) / cfg.horizontal_scale) for offset in (-1, 1)]
            origin = np.array([
                terrain_size[0] * 0.5 - border * cfg.horizontal_scale,
                terrain_size[1] * 0.5 - border * cfg.horizontal_scale,
                np.max(padded[x1:x2, y1:y2]) * cfg.vertical_scale,
            ])
            return heights, origin

        return lab_height_field_to_mesh(with_origin)(difficulty, cfg)

    return wrapper


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        mesh = meshes[0]
        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = [mesh]

        # Get mesh bounds
        bounds = mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # Left wall
        if np.random.uniform() < cfg.wall_prob[0]:
            left_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(left_wall)

        # Right wall
        if np.random.uniform() < cfg.wall_prob[1]:
            right_wall = trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            )
            result_meshes.append(right_wall)

        # Front wall
        if np.random.uniform() < cfg.wall_prob[2]:
            front_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(front_wall)

        # Back wall
        if np.random.uniform() < cfg.wall_prob[3]:
            back_wall = trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            )
            result_meshes.append(back_wall)

        return result_meshes, origin

    return wrapper
