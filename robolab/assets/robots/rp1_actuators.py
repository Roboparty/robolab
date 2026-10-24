# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RP1 delayed PD control with its measured motor torque-speed envelope."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


def dc_motor_envelope(
    joint_vel: torch.Tensor,
    saturation_effort: torch.Tensor,
    velocity_limit: torch.Tensor,
    max_motor_effort: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four-quadrant effort bounds, including the peak-torque plateau."""
    corner = velocity_limit * (1.0 + max_motor_effort / saturation_effort)
    velocity = joint_vel.clamp(min=-corner, max=corner)
    upper = torch.minimum(saturation_effort * (1.0 - velocity / velocity_limit), max_motor_effort)
    lower = torch.maximum(saturation_effort * (-1.0 - velocity / velocity_limit), -max_motor_effort)
    return lower, upper


class RP1Actuator(DelayedPDActuator):
    """Apply the same explicit, delayed motor dynamics on both physics backends."""

    cfg: RP1ActuatorCfg

    def __init__(self, cfg: RP1ActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._keep_velocity = resolve_joint_parameter(
            cfg.keep_velocity, None, self._joint_names, self._num_envs, self._device
        )
        self._rated_effort = resolve_joint_parameter(
            cfg.rated_effort, None, self._joint_names, self._num_envs, self._device
        )
        self._rated_velocity = resolve_joint_parameter(
            cfg.rated_velocity, None, self._joint_names, self._num_envs, self._device
        )
        if torch.any(self._keep_velocity <= 0.0) or torch.any(
            self._keep_velocity >= self.actuator_velocity_limit
        ):
            raise ValueError("RP1Actuator keep_velocity must lie within (0, actuator_velocity_limit).")
        if torch.any(self._rated_effort <= 0.0) or torch.any(
            self._rated_effort > self.actuator_effort_limit
        ):
            raise ValueError("RP1Actuator rated_effort must lie within (0, actuator_effort_limit].")
        if torch.any(self._rated_velocity <= 0.0) or torch.any(
            self._rated_velocity > self.actuator_velocity_limit
        ):
            raise ValueError("RP1Actuator rated_velocity must lie within (0, actuator_velocity_limit].")
        self._saturation_effort = self.actuator_effort_limit / (
            1.0 - self._keep_velocity / self.actuator_velocity_limit
        )

    @property
    def rated_effort(self) -> torch.Tensor:
        """Continuous rated joint effort, in N-m."""
        return self._rated_effort

    @property
    def rated_velocity(self) -> torch.Tensor:
        """Rated joint speed at the continuous operating point, in rad/s."""
        return self._rated_velocity

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor | None = None) -> torch.Tensor:
        if joint_vel is None:
            joint_vel = self._joint_vel
        lower, upper = dc_motor_envelope(
            joint_vel, self._saturation_effort, self.actuator_velocity_limit, self.actuator_effort_limit
        )
        return torch.clip(effort, min=lower, max=upper)


@configclass
class RP1ActuatorCfg(DelayedPDActuatorCfg):
    """Measured peak, plateau, and continuous operating-point parameters."""

    class_type: type = RP1Actuator
    keep_velocity: dict[str, float] | float = MISSING
    rated_effort: dict[str, float] | float = MISSING
    rated_velocity: dict[str, float] | float = MISSING
