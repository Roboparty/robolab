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


def _authored_height_field(func, difficulty, cfg):
    from .surface_mesh import heightfield_mesh, terrace_meshes

    if 0 < cfg.border_width < cfg.horizontal_scale:
        raise ValueError("Terrain border must be at least one grid cell.")
    shape = tuple(int(size / cfg.horizontal_scale) + 1 for size in cfg.size)
    border = int(cfg.border_width / cfg.horizontal_scale) + 1
    inner = cfg.copy()
    inner.size = tuple((n - 2 * border) * cfg.horizontal_scale for n in shape)
    heights = np.zeros(shape, dtype=np.int16)
    heights[border:-border, border:-border] = func(difficulty, inner)
    if func.__name__ == "perlin_pyramid_stairs_terrain":
        clean = inner.copy()
        clean.perlin_cfg = None
        levels = np.zeros(shape, dtype=np.int16)
        levels[border:-border, border:-border] = func(difficulty, clean)
        meshes = terrace_meshes(heights * cfg.vertical_scale, levels, cfg.horizontal_scale)
    else:
        meshes = [heightfield_mesh(heights * cfg.vertical_scale, cfg.horizontal_scale)]
    x1, y1 = (int((size * 0.5 - 1) / cfg.horizontal_scale) for size in cfg.size)
    x2, y2 = (int((size * 0.5 + 1) / cfg.horizontal_scale) for size in cfg.size)
    z = heights[x1:x2, y1:y2].max() * cfg.vertical_scale
    return meshes, np.array([cfg.size[0] / 2, cfg.size[1] / 2, z])


def height_field_to_mesh(func: Callable) -> Callable:
    """Preserve authored samples when requested; else Isaac Lab conversion with center origin."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        if getattr(cfg, "preserve_heightfield_samples", False):
            return _authored_height_field(func, difficulty, cfg)

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

        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = list(meshes)
        bounds = np.array([
            np.min([mesh.bounds[0] for mesh in meshes], axis=0),
            np.max([mesh.bounds[1] for mesh in meshes], axis=0),
        ])
        min_bound, max_bound = bounds[0], bounds[1]

        if np.random.uniform() < cfg.wall_prob[0]:
            result_meshes.append(trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [min_bound[0] - wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            ))
        if np.random.uniform() < cfg.wall_prob[1]:
            result_meshes.append(trimesh.creation.box(
                extents=[wall_thickness, max_bound[1] - min_bound[1], wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [max_bound[0] + wall_thickness / 2, (min_bound[1] + max_bound[1]) / 2, wall_height / 2]
                ),
            ))
        if np.random.uniform() < cfg.wall_prob[2]:
            result_meshes.append(trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, min_bound[1] - wall_thickness / 2, wall_height / 2]
                ),
            ))
        if np.random.uniform() < cfg.wall_prob[3]:
            result_meshes.append(trimesh.creation.box(
                extents=[max_bound[0] - min_bound[0], wall_thickness, wall_height],
                transform=trimesh.transformations.translation_matrix(
                    [(min_bound[0] + max_bound[0]) / 2, max_bound[1] + wall_thickness / 2, wall_height / 2]
                ),
            ))
        return result_meshes, origin

    return wrapper
