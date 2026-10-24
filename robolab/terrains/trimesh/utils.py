from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.terrains import SubTerrainBaseCfg


def crop_terrain_mesh_aabb(
    terrain_mesh: trimesh.Trimesh,
    x_max: float | None = None,
    x_min: float | None = None,
    y_max: float | None = None,
    y_min: float | None = None,
    z_max: float | None = None,
    z_min: float | None = None,
) -> trimesh.Trimesh:
    """Crop the terrain mesh to given the bounding box coordinates.
    Args:
        terrain_mesh (trimesh.Trimesh): The terrain mesh to be cropped.
        size (tuple[float, float]): The size of the bounding box (width, depth).
    Returns:
        trimesh.Trimesh: The cropped terrain mesh.
    """

    if x_max is not None:
        slice_plane_normal = [-1, 0, 0]
        slice_plane_origin = [x_max, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if x_min is not None:
        slice_plane_normal = [1, 0, 0]
        slice_plane_origin = [x_min, 0, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_max is not None:
        slice_plane_normal = [0, -1, 0]
        slice_plane_origin = [0, y_max, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if y_min is not None:
        slice_plane_normal = [0, 1, 0]
        slice_plane_origin = [0, y_min, 0]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_max is not None:
        slice_plane_normal = [0, 0, -1]
        slice_plane_origin = [0, 0, z_max]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    if z_min is not None:
        slice_plane_normal = [0, 0, 1]
        slice_plane_origin = [0, 0, z_min]
        terrain_mesh = trimesh.intersections.slice_mesh_plane(
            terrain_mesh,
            plane_normal=slice_plane_normal,
            plane_origin=slice_plane_origin,
        )
    return terrain_mesh


def generate_wall(func: Callable) -> Callable:
    """Wrapper to add walls to the generated terrain mesh."""

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: SubTerrainBaseCfg):
        meshes, origin = func(difficulty, cfg)
        if cfg is None or not hasattr(cfg, "wall_prob"):
            return meshes, origin

        wall_height = cfg.wall_height
        wall_thickness = cfg.wall_thickness
        result_meshes = meshes

        # Get mesh bounds
        mesh = max(meshes, key=lambda m: np.prod(m.bounds[1, :2] - m.bounds[0, :2]))
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
