from __future__ import annotations

import torch
import warp as wp
from functools import lru_cache

import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.utils.backend_utils import FactoryBase

from robolab.utils.warp.raycast import raycast_mesh_grouped


@lru_cache(maxsize=None)
def _backend_sensor_type(sensor_type: type, backend: str) -> type:
    # Task configs are imported before the simulator starts; load backend views at construction.
    if backend == "newton":
        from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import _LegacyNewtonRayCasterMixin

        mixin = _LegacyNewtonRayCasterMixin
    elif backend == "physx":
        from isaaclab_physx.sensors.ray_caster.ray_caster import _PhysXRayCasterMixin

        mixin = _PhysXRayCasterMixin
    else:
        raise ValueError(f"Unsupported ray-caster backend: {backend}")
    return type(f"{backend.title()}{sensor_type.__name__}", (mixin, sensor_type), {})


class _GroupedRayCastMixin:
    """Keep the grouped caster's near clipping on the 3.0 per-environment mesh tables."""

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        mesh_count = self._mesh_ids_wp.shape[1]
        self._ray_collision_groups = torch.arange(self._num_envs, dtype=torch.int32, device=self.device)
        self._ray_collision_groups = self._ray_collision_groups[:, None].expand(-1, self.num_rays)
        self._mesh_idxs_for_group = torch.arange(
            self._num_envs * mesh_count, dtype=torch.int32, device=self.device
        )
        self._meah_idxs_slice_for_group = torch.arange(
            self._num_envs + 1, dtype=torch.int32, device=self.device
        ) * mesh_count
        # Warp's Torch bridge accepts int64 storage for uint64 mesh handles.
        self._mesh_wp_ids = wp.to_torch(self._mesh_ids_wp).view(torch.int64).flatten()

    def _get_mesh_transforms_and_inv_transforms(self):
        positions = wp.to_torch(self._mesh_positions_w)
        orientations = wp.to_torch(self._mesh_orientations_w)
        inv_q = math_utils.quat_inv(orientations)
        inv_p = math_utils.quat_apply(inv_q, -positions)
        return torch.cat((positions, orientations), dim=-1).reshape(-1, 7), torch.cat(
            (inv_p, inv_q), dim=-1
        ).reshape(-1, 7)

    def _cast_grouped_rays(self, env_mask: wp.array, max_distance: float, return_distance=False, return_normal=False):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()
        transforms, inverse_transforms = self._get_mesh_transforms_and_inv_transforms()
        result = raycast_mesh_grouped(
            mesh_wp_device=wp.get_device(self.device),
            mesh_wp_ids=self._mesh_wp_ids,
            mesh_transforms=transforms,
            mesh_inv_transforms=inverse_transforms,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_idxs_for_group=self._mesh_idxs_for_group,
            meah_idxs_slice_for_group=self._meah_idxs_slice_for_group,
            ray_starts=wp.to_torch(self._ray_starts_w)[env_ids],
            ray_directions=wp.to_torch(self._ray_directions_w)[env_ids],
            max_dist=max_distance,
            min_dist=self.cfg.min_distance,
            return_distance=return_distance,
            return_normal=return_normal,
        )
        return env_ids, result


class GroupedRayCaster(_GroupedRayCastMixin, BaseMultiMeshRayCaster):
    """Ray caster that restricts each ray to its environment's meshes."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(_backend_sensor_type(cls, FactoryBase._get_backend()))

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids, (hits, _, _, _, _) = self._cast_grouped_rays(env_mask, self.cfg.max_distance)
        self._data.ray_hits_w.torch[env_ids] = hits
