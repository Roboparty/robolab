# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Left-right symmetry in the RPO observation and action spaces."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    The original batch is followed by its left-right mirror. Vector observations and
    the separate depth group retain their configured shapes.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2, *([1] * (obs.batch_dims - 1)))

        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env, obs["policy"], _joint_names(env)
        )
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(
            env, obs["critic"], _joint_names(env)
        )
        obs_aug["depth"][:batch_size] = obs["depth"][:]
        obs_aug["depth"][batch_size : 2 * batch_size] = _transform_depth_obs_left_right(obs["depth"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.empty((batch_size * 2, *actions.shape[1:]), device=actions.device, dtype=actions.dtype)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions, _joint_names(env))

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _height_scan_left_right_dims(env: ManagerBasedRLEnv) -> tuple[int, int, int]:
    """(history_length, ny, nx) from the running env cfg (dataclass fields are not on config *classes*)."""
    cfg = getattr(env, "unwrapped", env).cfg
    pat = cfg.scene.height_scanner.pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height_scan L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    hist = cfg.observations.critic.height_scan.history_length
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` (same arange endpoints / step).
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist, ny, nx


def _transform_height_scan_left_right(env: ManagerBasedRLEnv, hs: torch.Tensor) -> torch.Tensor:
    """Mirror lateral (y) grid rows; scalars only reorder (unlike ``depth_image``, scan is flat w.r.t. spatial axes)."""
    hist, ny, nx = _height_scan_left_right_dims(env)
    out = hs.reshape(*hs.shape[:-1], hist, ny, nx).flip(dims=[-2])
    return out.reshape(hs.shape)


def _history_length(env: ManagerBasedRLEnv, group_name: str) -> int:
    cfg = getattr(env, "unwrapped", env).cfg
    group = getattr(cfg.observations, group_name)
    return group.base_ang_vel.history_length


def _transform_policy_obs_left_right(
    env: ManagerBasedRLEnv, obs: torch.Tensor, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Mirror the concatenated Parkour policy vector in its configured term order."""
    history = _history_length(env, "policy")
    joint_dim = len(joint_names) if joint_names else 23
    obs = obs.clone()
    offset = 0
    for signs in ([-1, 1, -1], [1, -1, 1], [1, -1, -1]):
        width = history * 3
        obs[..., offset : offset + width] = _apply_xyz_sign(obs[..., offset : offset + width], signs)
        offset += width
    for _ in range(3):
        width = history * joint_dim
        obs[..., offset : offset + width] = _switch_joints_left_right_flat(
            obs[..., offset : offset + width], joint_names
        )
        offset += width
    return obs


def _transform_critic_obs_left_right(
    env: ManagerBasedRLEnv, obs: torch.Tensor, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Mirror the concatenated Parkour critic vector in its configured term order."""
    history = _history_length(env, "critic")
    joint_dim = len(joint_names) if joint_names else 23
    obs = obs.clone()
    offset = 0
    for signs in ([1, -1, 1], [-1, 1, -1], [1, -1, 1], [1, -1, -1]):
        width = history * 3
        obs[..., offset : offset + width] = _apply_xyz_sign(obs[..., offset : offset + width], signs)
        offset += width
    for _ in range(3):
        width = history * joint_dim
        obs[..., offset : offset + width] = _switch_joints_left_right_flat(
            obs[..., offset : offset + width], joint_names
        )
        offset += width
    height_width = obs.shape[-1] - offset
    obs[..., offset : offset + height_width] = _transform_height_scan_left_right(
        env, obs[..., offset : offset + height_width]
    )
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to depth image observations."""
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor, joint_names: list[str] | None = None) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, len(joint_names) if joint_names else 23)
    joint_data = _switch_joints_left_right(joint_data, joint_names)
    return joint_data.reshape(joint_data_shape)


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor, joint_names: list[str] | None = None) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:], joint_names)
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[           
'left_thigh_yaw_joint',   #0
'right_thigh_yaw_joint',  #1
'torso_joint',            #2
'left_thigh_roll_joint',  #3
'right_thigh_roll_joint', #4
'left_arm_pitch_joint',   #5
'right_arm_pitch_joint',  #6
'left_thigh_pitch_joint', #7
'right_thigh_pitch_joint',#8
'left_arm_roll_joint',    #9
'right_arm_roll_joint',   #10
'left_knee_joint',        #11
'right_knee_joint',       #12
'left_arm_yaw_joint',     #13
'right_arm_yaw_joint',    #14
'left_ankle_pitch_joint', #15
'right_ankle_pitch_joint',#16
'left_elbow_pitch_joint', #17
'right_elbow_pitch_joint',#18
'left_ankle_roll_joint',  #19
'right_ankle_roll_joint', #20
'left_elbow_yaw_joint',   #21
'right_elbow_yaw_joint'   #22
]

"""


def _joint_names(env: ManagerBasedRLEnv) -> list[str] | None:
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    return scene["robot"].joint_names if scene is not None else None


def _switch_joints_left_right(joint_data: torch.Tensor, joint_names: list[str] | None = None) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    if joint_names is not None:
        mirrored_names = [
            "right_" + name[5:] if name.startswith("left_") else
            "left_" + name[6:] if name.startswith("right_") else name
            for name in joint_names
        ]
        indices = torch.tensor([joint_names.index(name) for name in mirrored_names], device=joint_data.device)
        signs = torch.tensor([
            -1 if name == "torso_joint" or name.endswith(("_yaw_joint", "_roll_joint")) else 1
            for name in joint_names
        ], device=joint_data.device, dtype=joint_data.dtype)
        return joint_data[..., indices] * signs
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = joint_data[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
    
    joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] = -1 * joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]]

    return joint_data_switched
