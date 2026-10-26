"""Sub-module containing Flat-Patch based velocity command generators."""

from __future__ import annotations

import math

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

def sample_uniform(ranges: torch.Tensor) -> torch.Tensor:
    """Sample one value uniformly from each [low, high] range."""
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("ranges must have shape (batch, 2).")
    if torch.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("range upper bounds must be greater than or equal to lower bounds.")
    return ranges[:, 0] + torch.rand(
        len(ranges), device=ranges.device, dtype=ranges.dtype
    ) * (ranges[:, 1] - ranges[:, 0])


def clamp_to_ranges(values: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Clamp each scalar value to its corresponding lower and upper bounds."""
    if values.ndim != 1 or ranges.shape != (len(values), 2):
        raise ValueError("values and ranges must have shapes (batch,) and (batch, 2).")
    if torch.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("range upper bounds must be greater than or equal to lower bounds.")
    return torch.maximum(torch.minimum(values, ranges[:, 1]), ranges[:, 0])


def target_distance_reached(
    target_xy: torch.Tensor,
    root_xy: torch.Tensor,
    distance_threshold: float,
) -> torch.Tensor:
    """Return whether each root is within the inclusive distance threshold of its target."""
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    return torch.norm(target_xy - root_xy, dim=1) <= distance_threshold


def signed_target_heading_offset(
    axis: torch.Tensor,
    signed_speed: torch.Tensor,
) -> torch.Tensor:
    """Return the body-heading offset that points the selected signed body axis at a target.

    Axis values are ``1`` for body x and ``2`` for body y. A zero speed has no
    offset; the caller should hold the current heading while that command is idle.
    """
    if axis.shape != signed_speed.shape:
        raise ValueError("axis and signed_speed must have matching shapes.")
    if torch.any((axis != 1) & (axis != 2)):
        raise ValueError("axis values must be 1 (x) or 2 (y).")

    offset = torch.zeros_like(signed_speed)
    offset[(axis == 1) & (signed_speed < 0.0)] = math.pi
    offset[(axis == 2) & (signed_speed > 0.0)] = -0.5 * math.pi
    offset[(axis == 2) & (signed_speed < 0.0)] = 0.5 * math.pi
    return offset


def signed_target_linear_velocity(
    axis: torch.Tensor,
    signed_speed: torch.Tensor,
    target_distance: torch.Tensor,
    velocity_stiffness: float,
) -> torch.Tensor:
    """Compute axis-aligned signed commands with distance-based magnitude."""
    if not (
        axis.shape
        == signed_speed.shape
        == target_distance.shape
    ):
        raise ValueError("signed-target inputs must have matching shapes.")
    if torch.any((axis != 1) & (axis != 2)):
        raise ValueError("axis values must be 1 (x) or 2 (y).")
    if velocity_stiffness < 0.0:
        raise ValueError("velocity_stiffness must be non-negative.")

    magnitude = torch.minimum(
        target_distance.clamp_min(0.0) * velocity_stiffness,
        signed_speed.abs(),
    )
    command = torch.zeros(len(axis), 2, device=axis.device, dtype=signed_speed.dtype)
    command[axis == 1, 0] = signed_speed[axis == 1].sign() * magnitude[axis == 1]
    command[axis == 2, 1] = signed_speed[axis == 2].sign() * magnitude[axis == 2]
    return command


def select_distant_target_indices(
    targets_xy: torch.Tensor,
    root_xy: torch.Tensor,
    distance_threshold: float,
    random_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose a random target beyond the threshold, or the farthest if none qualify."""
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    if targets_xy.ndim != 3 or targets_xy.shape[-1] != 2:
        raise ValueError("targets_xy must have shape (batch, targets, 2).")
    if root_xy.shape != (targets_xy.shape[0], 2):
        raise ValueError("root_xy must have shape (batch, 2).")
    if targets_xy.shape[1] == 0:
        raise ValueError("targets_xy must contain at least one candidate per batch.")
    if random_scores is None:
        random_scores = torch.rand(
            targets_xy.shape[:2], device=targets_xy.device, dtype=targets_xy.dtype
        )
    elif random_scores.shape != targets_xy.shape[:2]:
        raise ValueError("random_scores must have shape (batch, targets).")

    distances = torch.norm(targets_xy - root_xy.unsqueeze(1), dim=-1)
    valid = distances > distance_threshold
    random_choice = random_scores.masked_fill(~valid, float("-inf")).argmax(dim=1)
    farthest_choice = distances.argmax(dim=1)
    return torch.where(valid.any(dim=1), random_choice, farthest_choice)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import PoseVelocityCommandCfg


class PoseVelocityCommand(CommandTerm):
    """Velocity command based on the 2D flat patch command generator."""

    cfg: PoseVelocityCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PoseVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        # Per-resample linear-speed limits. Yaw is computed from the heading error
        # and clamped directly by the terrain-specific angular-velocity range.
        self.max_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_exp_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        self.lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)

        self.random_lin_vel_x_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_lin_vel_y_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_ang_vel_z_range = torch.zeros(self.num_envs, 2, device=self.device)
        self.random_velocity_indices = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.random_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.random_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.random_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        # 0: regular target control, 1: signed body-x control, 2: signed body-y control.
        self.signed_target_axis = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        self.signed_target_speed = torch.zeros(self.num_envs, device=self.device)
        # default random-velocity range (used as fallback for terrains without a dedicated entry below)
        self.random_lin_vel_x_range[:, 0] = self.cfg.ranges.lin_vel_x[0]
        self.random_lin_vel_x_range[:, 1] = self.cfg.ranges.lin_vel_x[1]
        self.random_lin_vel_y_range[:, 0] = self.cfg.ranges.lin_vel_y[0]
        self.random_lin_vel_y_range[:, 1] = self.cfg.ranges.lin_vel_y[1]
        self.random_ang_vel_z_range[:, 0] = self.cfg.ranges.ang_vel_z[0]
        self.random_ang_vel_z_range[:, 1] = self.cfg.ranges.ang_vel_z[1]

        if self.cfg.signed_target_velocity_axes and self.cfg.velocity_ranges is None:
            raise ValueError("signed_target_velocity_axes requires velocity_ranges.")

        if self.cfg.velocity_ranges is not None:
            terrain_generator_cfg = self.terrain.cfg.terrain_generator
            # Column → sub-terrain mapping from the generated grid (supports one-col-per-type).
            terrain_gen = getattr(self.terrain, "terrain_generator", None)
            grid = getattr(terrain_gen, "subterrain_index_grid", None) if terrain_gen is not None else None
            if grid is not None:
                sub_indices = np.asarray(grid[0], dtype=np.int32)
            else:
                proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()])
                proportions /= np.sum(proportions)
                sub_indices = []
                for index in range(terrain_generator_cfg.num_cols):
                    sub_index = np.min(
                        np.where(index / terrain_generator_cfg.num_cols + 0.001 < np.cumsum(proportions))[0]
                    )
                    sub_indices.append(sub_index)
                sub_indices = np.array(sub_indices, dtype=np.int32)
            sub_terrains_names = list(terrain_generator_cfg.sub_terrains.keys())
            for key, value in self.cfg.velocity_ranges.items():
                if key in sub_terrains_names:
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                        self.lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                        self.lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                        self.lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                        self.ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                        self.ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]
                else:
                    raise RuntimeError(f"Terrain type {key} not found in the terrain generator sub-terrain names.")

            if self.cfg.random_velocity_terrain is not None:
                for key in self.cfg.random_velocity_terrain:
                    if key not in sub_terrains_names:
                        raise RuntimeError(
                            f"Random-velocity terrain type {key} not found in the terrain generator."
                        )
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    # use the terrain-specific velocity range (falls back to the global `ranges` if not provided)
                    value = self.cfg.velocity_ranges.get(key)
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        self.random_velocity_indices[env_indices] = True
                        if value is not None:
                            self.random_lin_vel_x_range[env_indices, 0] = value["lin_vel_x"][0]
                            self.random_lin_vel_x_range[env_indices, 1] = value["lin_vel_x"][1]
                            self.random_lin_vel_y_range[env_indices, 0] = value["lin_vel_y"][0]
                            self.random_lin_vel_y_range[env_indices, 1] = value["lin_vel_y"][1]
                            self.random_ang_vel_z_range[env_indices, 0] = value["ang_vel_z"][0]
                            self.random_ang_vel_z_range[env_indices, 1] = value["ang_vel_z"][1]

            if self.cfg.signed_target_velocity_axes is not None:
                axis_values = {"x": 1, "y": 2}
                for key, axis_name in self.cfg.signed_target_velocity_axes.items():
                    if key not in sub_terrains_names:
                        raise RuntimeError(
                            f"Signed-target terrain type {key} not found in the terrain generator."
                        )
                    if axis_name not in axis_values:
                        raise ValueError(
                            f"Signed-target axis for {key} must be 'x' or 'y', got {axis_name!r}."
                        )
                    if key not in self.cfg.velocity_ranges:
                        raise ValueError(
                            f"Signed-target terrain {key} requires a terrain-specific velocity range."
                        )
                    terrain_type_index = sub_terrains_names.index(key)
                    type_indices = np.where(sub_indices == terrain_type_index)[0]
                    for type_indice in type_indices:
                        env_indices = torch.where(self.terrain.terrain_types == type_indice)[0]
                        if torch.any(self.random_velocity_indices[env_indices]):
                            raise ValueError(
                                f"Terrain {key} cannot use random and signed-target velocity control together."
                            )
                        self.signed_target_axis[env_indices] = axis_values[axis_name]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        lin_vel_error = torch.sum(
            torch.square(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        tracking_exp_vel_xy = torch.exp(
            -lin_vel_error / self.cfg.lin_vel_metrics_std**2
        )
        self.metrics["tracking_exp_vel_xy"] += (
            tracking_exp_vel_xy / self._env.max_episode_length
        )
        angular_vel_error = torch.square(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["tracking_exp_vel_yaw"] += (
            torch.exp(-angular_vel_error / self.cfg.ang_vel_metrics_std**2) / self._env.max_episode_length
        )

    def _resample_command(self, env_ids: Sequence[int] | slice | None):
        # IsaacLab 3.0 reset selector -> index tensor
        if env_ids is None or env_ids == slice(None):
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        elif torch.is_tensor(env_ids):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Select a random patch outside the reached radius. If no such patch
        # exists, the farthest patch still guarantees the best available target.
        candidate_targets = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]
        target_ids = select_distant_target_indices(
            candidate_targets[:, :, :2],
            self.robot.data.root_pos_w[env_ids, :2],
            self.cfg.target_dis_threshold,
        )
        batch_ids = torch.arange(len(env_ids), device=self.device)
        self.pos_command_w[env_ids] = candidate_targets[batch_ids, target_ids]

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.max_command_b[env_ids, 0] = self.lin_vel_x_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_x_range[env_ids, 1] - self.lin_vel_x_range[env_ids, 0]
        )
        # -- linear velocity - y direction
        self.max_command_b[env_ids, 1] = self.lin_vel_y_range[env_ids, 0] + r.uniform_(0.0, 1.0) * (
            self.lin_vel_y_range[env_ids, 1] - self.lin_vel_y_range[env_ids, 0]
        )
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Only update random velocities for envs that are currently being resampled AND are marked for random velocity
        # Create a mask for the current batch of env_ids
        current_batch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        current_batch_mask[env_ids] = True

        # Find intersection: envs in this batch that are also random velocity envs
        update_mask = current_batch_mask & self.random_velocity_indices
        random_velocity_env_ids = update_mask.nonzero(as_tuple=False).flatten()

        self._resample_random_velocity(random_velocity_env_ids)

        signed_target_env_ids = (
            current_batch_mask & (self.signed_target_axis != 0)
        ).nonzero(as_tuple=False).flatten()
        if signed_target_env_ids.numel() > 0:
            use_x = self.signed_target_axis[signed_target_env_ids] == 1
            speed_range = torch.where(
                use_x.unsqueeze(1),
                self.lin_vel_x_range[signed_target_env_ids],
                self.lin_vel_y_range[signed_target_env_ids],
            )
            self.signed_target_speed[signed_target_env_ids] = sample_uniform(speed_range)

    def _resample_random_velocity(self, env_ids: torch.Tensor):
        """Sample terrain-specific random velocity commands for the selected environments."""
        if env_ids.numel() == 0:
            return
        count = len(env_ids)
        self.random_lin_vel_x[env_ids] = self.random_lin_vel_x_range[env_ids, 0] + torch.rand(
            count, device=self.device
        ) * (self.random_lin_vel_x_range[env_ids, 1] - self.random_lin_vel_x_range[env_ids, 0])
        self.random_lin_vel_y[env_ids] = self.random_lin_vel_y_range[env_ids, 0] + torch.rand(
            count, device=self.device
        ) * (self.random_lin_vel_y_range[env_ids, 1] - self.random_lin_vel_y_range[env_ids, 0])
        self.random_ang_vel_z[env_ids] = self.random_ang_vel_z_range[env_ids, 0] + torch.rand(
            count, device=self.device
        ) * (self.random_ang_vel_z_range[env_ids, 1] - self.random_ang_vel_z_range[env_ids, 0])

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        target_dist = torch.norm(target_vec[:, :2], dim=1)

        reached_target = (
            ~self.random_velocity_indices
            & ~self.is_standing_env
            & target_distance_reached(
                self.pos_command_w[:, :2],
                self.robot.data.root_pos_w[:, :2],
                self.cfg.target_dis_threshold,
            )
        )
        reached_env_ids = reached_target.nonzero(as_tuple=False).flatten()
        if reached_env_ids.numel() > 0:
            self._resample(reached_env_ids)
            target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
            target_dist = torch.norm(target_vec[:, :2], dim=1)

        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), target_vec)

        # Set the desired body heading. Signed target terrains point their
        # selected positive or negative body axis toward the target.
        target_vec = self.pos_command_w - self.robot.data.root_pos_w
        target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        desired_heading = target_direction.clone()
        signed_target_indices = self.signed_target_axis != 0
        signed_target_env_ids = signed_target_indices.nonzero(as_tuple=False).flatten()
        if signed_target_env_ids.numel() > 0:
            signed_speed = self.signed_target_speed[signed_target_env_ids]
            desired_heading[signed_target_env_ids] += signed_target_heading_offset(
                self.signed_target_axis[signed_target_env_ids], signed_speed
            )
            idle_ids = signed_target_env_ids[signed_speed == 0.0]
            desired_heading[idle_ids] = self.robot.data.heading_w[idle_ids]

        # compute errors to find the closest direction to the current heading
        # this is done to avoid the discontinuity at the -pi/pi boundary
        self.heading_command_w = wrap_to_pi(desired_heading - self.robot.data.heading_w)

        raw_yaw_command = self.heading_command_w * self.cfg.heading_control_stiffness
        self.vel_command_b[:, 2] = clamp_to_ranges(raw_yaw_command, self.ang_vel_z_range)

        raw_velocity_b = self.pos_command_b[:, :2] * self.cfg.velocity_control_stiffness
        vx = raw_velocity_b[:, 0]
        vy = raw_velocity_b[:, 1]
        min_x = (
            -self.max_command_b[:, 0]
            if not self.cfg.only_positive_lin_vel_x
            else torch.zeros_like(self.max_command_b[:, 0])
        )
        min_y = -self.max_command_b[:, 1]
        max_x = self.max_command_b[:, 0]
        max_y = self.max_command_b[:, 1]
        eps = 1e-6

        abs_vx = vx.abs()
        abs_vy = vy.abs()

        if not self.cfg.only_positive_lin_vel_x:
            # clamp each axis independently
            clamped_vx = torch.clamp(abs_vx, min=min_x, max=max_x)
            clamped_vy = torch.clamp(abs_vy, min=min_y, max=max_y)

            # compute scale for whichever axis is dominant
            scale_x = clamped_vx / (abs_vx + eps)
            scale_y = clamped_vy / (abs_vy + eps)
            scale = torch.where(abs_vx >= abs_vy, scale_x, scale_y)

            # apply scale and restore sign
            self.vel_command_b[:, 0] = vx * scale
            self.vel_command_b[:, 1] = vy * scale

        else:
            max_forward_speed = torch.clamp(self.max_command_b[:, 0], min=0.0)
            forward_speed = torch.minimum(torch.clamp(vx, min=0.0), max_forward_speed)
            self.vel_command_b[:, 0] = forward_speed
            self.vel_command_b[:, 1] = 0.0

        if signed_target_env_ids.numel() > 0:
            self.vel_command_b[signed_target_env_ids, :2] = signed_target_linear_velocity(
                self.signed_target_axis[signed_target_env_ids],
                self.signed_target_speed[signed_target_env_ids],
                target_dist[signed_target_env_ids],
                self.cfg.velocity_control_stiffness,
            )

        self.vel_command_b[:] *= (target_dist > self.cfg.target_dis_threshold).unsqueeze(-1)
        self.vel_command_b[:, :2] *= (
            (torch.norm(self.vel_command_b[:, :2], dim=1) > self.cfg.lin_vel_threshold).float().unsqueeze(-1)
        )
        self.vel_command_b[:, 2] *= (torch.abs(self.vel_command_b[:, 2]) > self.cfg.ang_vel_threshold).float()

        random_velocity_env_ids = self.random_velocity_indices.nonzero(as_tuple=False).flatten()
        self.vel_command_b[random_velocity_env_ids, 0] = self.random_lin_vel_x[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 1] = self.random_lin_vel_y[random_velocity_env_ids]
        self.vel_command_b[random_velocity_env_ids, 2] = self.random_ang_vel_z[random_velocity_env_ids]

        # standing envs must stay zero even if they also fall on a random-velocity terrain.
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "flat_patch_visualizer"):
                # -- pose
                self.cfg.flat_patch_visualizer_cfg.markers["Goal"].radius = self.cfg.target_dis_threshold
                self.cfg.flat_patch_visualizer_cfg.markers["Patches"].radius = self.cfg.target_dis_threshold
                self.flat_patch_visualizer = VisualizationMarkers(self.cfg.flat_patch_visualizer_cfg)
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.flat_patch_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "flat_patch_visualizer"):
                self.flat_patch_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if getattr(self.cfg, "patch_vis", True):
            flat_patches = self.valid_targets.reshape(-1, 3)
            poses = torch.cat([self.pos_command_w, flat_patches], dim=0)
            marker_indices = torch.cat(
                [
                    torch.zeros(self.num_envs, dtype=torch.int, device=self.device),
                    torch.ones(flat_patches.shape[0], dtype=torch.int, device=self.device),
                ],
                dim=0,
            )
            self.flat_patch_visualizer.visualize(poses, marker_indices=marker_indices)
        else:
            marker_indices = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
            self.flat_patch_visualizer.visualize(self.pos_command_w, marker_indices=marker_indices)
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 8.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
