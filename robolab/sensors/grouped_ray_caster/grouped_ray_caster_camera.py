from __future__ import annotations

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from isaaclab.utils.backend_utils import FactoryBase

from .grouped_ray_caster import _GroupedRayCastMixin, _backend_sensor_type


class GroupedRayCasterCamera(_GroupedRayCastMixin, BaseMultiMeshRayCasterCamera):
    """Grouped ray camera preserving the original near and far clipping."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(_backend_sensor_type(cls, FactoryBase._get_backend()))

    def _update_buffers_impl(self, env_mask: wp.array):
        self._update_frame(env_mask, frame_op=1)
        env_ids, (hits, depth, normals, _, _) = self._cast_grouped_rays(
            env_mask, self.cfg.max_distance * 2, return_distance=True, return_normal=True
        )
        self.ray_hits_w.torch[env_ids] = hits
        if "distance_to_image_plane" in self.cfg.data_types:
            orientation = self._data.quat_w_world.torch[env_ids, None, :].expand(-1, self.num_rays, -1)
            directions = wp.to_torch(self._ray_directions_w)[env_ids]
            plane_depth = math_utils.quat_apply(math_utils.quat_inv(orientation), depth[..., None] * directions)[
                ..., 0
            ]
            if self.cfg.depth_clipping_behavior == "max":
                plane_depth = torch.nan_to_num(plane_depth.clamp(max=self.cfg.max_distance), nan=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                plane_depth = torch.where(plane_depth > self.cfg.max_distance, 0.0, plane_depth)
                plane_depth = torch.nan_to_num(plane_depth, nan=0.0)
            self._data.output["distance_to_image_plane"].torch[env_ids] = plane_depth.reshape(-1, *self.image_shape, 1)

        if "distance_to_camera" in self.cfg.data_types:
            if self.cfg.depth_clipping_behavior == "max":
                depth = depth.clamp(max=self.cfg.max_distance)
            elif self.cfg.depth_clipping_behavior == "zero":
                depth = torch.where(depth > self.cfg.max_distance, 0.0, depth)
            self._data.output["distance_to_camera"].torch[env_ids] = depth.reshape(-1, *self.image_shape, 1)

        if "normals" in self.cfg.data_types:
            self._data.output["normals"].torch[env_ids] = normals.reshape(-1, *self.image_shape, 3)
