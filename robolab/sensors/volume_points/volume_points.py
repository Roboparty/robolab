from __future__ import annotations

import re
import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pxr import UsdPhysics


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sim.utils.queries import path_expr_to_glob, resolve_matching_prims_from_source, split_path_expr
from isaaclab.utils.backend_utils import FactoryBase

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    import omni.physics.tensors as physx
    from isaaclab.markers import VisualizationMarkersCfg

    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(SensorBase):
    """Volume Points sensor for detecting volume points in a simulation."""

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)

        # Initialize the volume points
        self._volume_points = None
        self._physics_backend = FactoryBase._get_backend()
        self._body_physx_view = None
        if self._physics_backend == "newton":
            from isaaclab_newton.physics import NewtonManager

            self._newton_site_label = NewtonManager.cl_register_site(cfg.prim_path, wp.transform())

    """
    Properties
    """

    @property
    def data(self) -> VolumePointsData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with volume points sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with volume points sensors attached."""
        return self._body_names

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the rigid bodies captured (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._body_physx_view

    """
    Operations
    """

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        """Record the edges buffer to the sensor so that the penetration data can be updated.

        NOTE: typically this is called by a startup event.

        """
        self._virtual_obstacles.update(virtual_obstacles)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timers and counters
        super().reset(env_ids=env_ids, env_mask=env_mask)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Implementation
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        if self._physics_backend == "newton":
            self._initialize_newton_bodies()
        elif self._physics_backend == "physx":
            self._initialize_physx_bodies()
        else:
            raise ValueError(f"Unsupported volume-points backend: {self._physics_backend}")

        self._volume_points_pattern = self.cfg.points_generator.func(self.cfg.points_generator).to(self.device)
        self._data = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )
        self._virtual_obstacles = {}

    def _initialize_physx_bodies(self):
        from isaaclab_physx.physics import PhysxManager

        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        *parent_segments, leaf_pattern = split_path_expr(self.cfg.prim_path)
        body_pattern = re.compile(leaf_pattern)
        body_matches = resolve_matching_prims_from_source(
            "/".join(parent_segments),
            predicate=lambda prim: bool(body_pattern.fullmatch(prim.GetName())) and prim.HasAPI(UsdPhysics.RigidBodyAPI),
        )
        self._body_names = [prim.GetName() for prim, _ in body_matches]
        body_path_globs = [path_expr_to_glob(expr) for _, expr in body_matches]
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_path_globs)

        # resolve the true count of bodies
        self._num_bodies = self.body_physx_view.count // self._num_envs
        # check that volume points sensor succeeded
        if self._num_bodies != len(self._body_names):
            raise RuntimeError(
                "Failed to initialize volume points sensor for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_path_globs}"
            )

    def _initialize_newton_bodies(self):
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        global_index, per_world = NewtonManager._cl_site_index_map[self._newton_site_label]
        site_indices = [[global_index] for _ in range(self._num_envs)] if per_world is None else per_world
        self._num_bodies = len(site_indices[0])
        if not self._num_bodies or len(site_indices) != self._num_envs or any(
            len(indices) != self._num_bodies for indices in site_indices
        ):
            raise RuntimeError(f"Volume-points bodies do not match across environments: {self.cfg.prim_path}")
        site_ids = torch.tensor(site_indices, dtype=torch.long, device=self.device)
        self._newton_body_indices = wp.to_torch(model.shape_body)[site_ids].long()
        if (self._newton_body_indices < 0).any():
            raise RuntimeError(f"Volume-points sensor requires dynamic rigid bodies: {self.cfg.prim_path}")
        self._body_names = [model.body_label[index].rsplit("/", 1)[-1] for index in self._newton_body_indices[0].tolist()]

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # update the volume points data
        self._refresh_volume_points(env_ids)
        # update the penetration depth data
        self._refresh_penetration_offset(env_ids)

    def _refresh_volume_points(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh the volume points data. If env_ids is None, refresh all environments."""

        if env_ids is None:
            env_ids = slice(None)
        if getattr(self, "_physics_backend", "physx") == "newton":
            from isaaclab_newton.physics import NewtonManager

            NewtonManager.forward()
            state = NewtonManager.get_state_0()
            model = NewtonManager.get_model()
            body_ids = self._newton_body_indices[env_ids]
            body_poses = wp.to_torch(state.body_q)[body_ids]
            body_vels = wp.to_torch(state.body_qd)[body_ids].clone()
            com_b = wp.to_torch(model.body_com)[body_ids]
            com_w = math_utils.quat_apply(body_poses[..., 3:], com_b)
            body_vels[..., :3] -= torch.linalg.cross(body_vels[..., 3:], com_w, dim=-1)
        else:
            # A list of body patterns creates a pattern-major PhysX view (body, env).
            body_poses = wp.to_torch(self.body_physx_view.get_transforms()).view(self.num_bodies, self._num_envs, 7)
            body_vels = wp.to_torch(self.body_physx_view.get_velocities()).view(self.num_bodies, self._num_envs, 6)
            body_poses = body_poses.transpose(0, 1)[env_ids]
            body_vels = body_vels.transpose(0, 1)[env_ids]
        self._data.pos_w[env_ids] = body_poses[..., :3]  # (N_, B, 3)
        self._data.quat_w[env_ids] = body_poses[..., 3:]  # (N_, B, 4), xyzw
        self._data.vel_w[env_ids] = body_vels[..., :3]  # (N_, B, 3)
        self._data.ang_vel_w[env_ids] = body_vels[..., 3:]  # (N_, B, 3)

        # calculate the volume points positions and velocities in world frame
        N_B = self._data.pos_w[env_ids].shape[0] * self._data.pos_w[env_ids].shape[1]  # (N_*B)
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(N_B, -1, -1),  # (N_*B, P, 3)
            self._data.pos_w[env_ids].flatten(0, 1),  # (N_*B, 3)
            self._data.quat_w[env_ids].flatten(0, 1),  # (N_*B, 4)
        ).reshape(
            *self._data.pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3
        )  # (N_, B, P, 3)
        self._data.points_pos_w[env_ids] = points_pos_w
        points_vel_w = self._data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()  # (N_, B, P, 3)
        points_vel_w += torch.linalg.cross(
            self._data.ang_vel_w[env_ids].unsqueeze(-2),
            (points_pos_w - self._data.pos_w[env_ids].unsqueeze(-2)),
            dim=-1,
        )  # (N_, B, P, 3)
        self._data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: Sequence[int]) -> None:
        """Refresh the penetration depth data. If env_ids is None, refresh all environments."""

        penetration_offset_buf: torch.Tensor = self._data.penetration_offset[env_ids]
        penetration_offset_buf[:] = 0.0
        penetration_depth_buf = torch.zeros_like(penetration_offset_buf[..., 0])  # (N_, B, P)

        for virtual_obstacle in self._virtual_obstacles.values():
            # get the penetration offset for the virtual obstacle
            penetration_offset = virtual_obstacle.get_points_penetration_offset(
                self._data.points_pos_w[env_ids].flatten(0, 2)
            )  # (N_*B*P, 3)
            penetration_offset = penetration_offset.reshape(self._data.points_pos_w[env_ids].shape)  # (N_, B, P, 3)
            penetration_depth = torch.norm(penetration_offset, dim=-1)  # (N_, B, P)
            # update the penetration offset if the depth is greater than the current depth
            mask = penetration_depth > penetration_depth_buf  # (N_, B, P)
            penetration_depth_buf[mask] = penetration_depth[mask]
            penetration_offset_buf[mask] = penetration_offset[mask]

        self._data.penetration_offset[env_ids] = penetration_offset_buf

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.points_visualizer.set_visibility(True)
        else:
            if hasattr(self, "points_visualizer"):
                self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if not self.is_initialized:
            return

        points = self._data.points_pos_w.view(-1, 3)  # (N_*B*P, 3)
        penetrated = torch.norm(self._data.penetration_offset.view(-1, 3), dim=-1) > 0.0  # (N_*B*P,)

        # add penetrated points if none
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=self.device)], dim=0)

        self.points_visualizer.visualize(
            translations=points,
            marker_indices=penetrated.long(),
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        if hasattr(self, "point_visualizer"):
            delattr(self, "points_visualizer")
        self._physics_sim_view = None
        self._body_physx_view = None
