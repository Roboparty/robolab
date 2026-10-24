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

import numpy as np
import torch
from typing import Optional

import isaaclab.utils.math as math_utils

def quaternion_to_xyzw(quat: torch.Tensor, source_order: str) -> torch.Tensor:
    """Convert serialized motion quaternions to Isaac Lab's runtime convention."""
    if source_order not in ("wxyz", "xyzw"):
        raise ValueError(f"Unsupported quaternion order: {source_order!r}.")
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternions ending in 4 components, got {quat.shape}.")
    return math_utils.convert_quat(quat, to="xyzw") if source_order == "wxyz" else quat

@torch.jit.script
def vel_forward_diff(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the forward differences of the input data

    Args:
        data (torch.Tensor): The input data tensor of shape (N, dim).
        dt (float): The time step duration.
    """
    N = data.shape[0]
    if N < 2:
        raise RuntimeError(f"Input data has only {N} frames, cannot compute velocity.")
    vel = torch.zeros_like(data)
    vel[:-1] = (data[1:] - data[:-1]) / dt
    vel[-1] = vel[-2]  # use the last value as the same as the second last value
    return vel


@torch.jit.script
def ang_vel_from_quat_diff(quat: torch.Tensor, dt: float, in_frame:str = "body") -> torch.Tensor:
    """Compute the angular velocity from quaternion differences.

    Args:
        quat (torch.Tensor): The input quaternion tensor of shape (N, 4), 
                            representing the rotation from world to body frame.
        dt (float): The time step duration.
        in_frame (str): The frame in which the angular velocity is expressed, either "body" or "world".
    """
    if in_frame not in ["body", "world"]:
        raise ValueError(f"Invalid in_frame value: {in_frame}. Must be 'body' or 'world'.")
    
    N = quat.shape[0]
    if N < 2:
        raise RuntimeError(f"Input quaternion has only {N} frames, cannot compute angular velocity.")
    
    ang_vel = torch.zeros((N, 3), dtype=torch.float32, device=quat.device)
    for i in range(N-1):
        q1 = quat[i].unsqueeze(0)  # from world frame to body, shape (1, 4)
        q2 = quat[i + 1].unsqueeze(0)  # from world frame to body (at next time), shape (1, 4)

        diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q1), q2)
        diff_angle_axis = math_utils.axis_angle_from_quat(diff_quat)
        if in_frame == "world":
            diff_angle_axis = math_utils.quat_apply(q1, diff_angle_axis)
        ang_vel[i, :] = diff_angle_axis.squeeze() / dt  # convert to angular velocity

    ang_vel[-1, :] = ang_vel[-2, :]  # use the last value as the same as the second last value
    
    return ang_vel


def quat_slerp(
    q0: torch.Tensor,
    *,
    q1: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Interpolation between consecutive rotations (Spherical Linear Interpolation).

    Args:
        q0: The first quaternion (xyzw). Shape is (N, 4) or (N, M, 4).
        q1: The second quaternion (xyzw). Shape is (N, 4) or (N, M, 4).
        blend: Interpolation coefficient between 0 (q0) and 1 (q1). Shape is (N,) or (N, M).
        start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
        end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

    Returns:
        Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
    """
    if start is not None and end is not None:
        return quat_slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q1 is None or blend is None:
        raise ValueError("Quaternion interpolation requires q1 and blend.")
    while blend.ndim < q0.ndim:
        blend = blend.unsqueeze(-1)

    q0 = math_utils.normalize(q0)
    q1 = math_utils.normalize(q1)
    cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    q1 = torch.where(cos_half_theta < 0, -q1, q1)
    cos_half_theta = cos_half_theta.abs()
    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt((1.0 - cos_half_theta.square()).clamp(min=0.0))
    small_angle = sin_half_theta < 0.001
    denominator = torch.where(small_angle, torch.ones_like(sin_half_theta), sin_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / denominator
    ratio_b = torch.sin(blend * half_theta) / denominator
    spherical = ratio_a * q0 + ratio_b * q1
    linear = (1 - blend) * q0 + blend * q1
    return math_utils.normalize(torch.where(small_angle, linear, spherical))

@torch.jit.script
def linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Linear interpolate between two tensors.

    Args:
        x0 (torch.Tensor): shape (N, M)
        x1 (torch.Tensor): shape (N, M)
        blend (torch.Tensor): shape(N, 1)
    """
    return x0 * (1 - blend) + x1 * blend


@torch.jit.script
def calc_frame_blend(time:torch.Tensor, duration:torch.Tensor, num_frames:torch.Tensor, dt:torch.Tensor):

    phase = time / duration
    phase = torch.clamp(phase, min=0.0, max=1.0)
    
    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0.float() * dt) / dt
    
    return frame_idx0, frame_idx1, blend


@torch.jit.script
def rotmat_to_euler_yzx(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in x(q0)z(q1)y(q2) sequence
    x = torch.atan2(mat[:, 2, 1], mat[:, 1, 1])  # x
    z = torch.asin(-mat[:, 0, 1])  # z
    y = torch.atan2(mat[:, 0, 2], mat[:, 0, 0])  # y
    x = math_utils.wrap_to_pi(x)
    z = math_utils.wrap_to_pi(z)
    y = math_utils.wrap_to_pi(y)
    return y, z, x


@torch.jit.script
def rotmat_to_euler_xzy(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in y(q0)z(q1)x(q2) sequence
    y = torch.atan2(-mat[:, 2, 0], mat[:, 0, 0])  # y
    z = torch.asin(mat[:, 1, 0])  # z
    x = torch.atan2(-mat[:, 1, 2], mat[:, 1, 1])  # x
    y = math_utils.wrap_to_pi(y)
    z = math_utils.wrap_to_pi(z)
    x = math_utils.wrap_to_pi(x)
    return x, z, y


def zxy_to_xyz(points):
    """Convert the points from y-up to z-up."""
    return points[..., [2, 0, 1]]


def xyz_to_zxy(points):
    """Convert the points from z-up to y-up."""
    return points[..., [1, 2, 0]]


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to tangent and normal vectors. (Directly copied from MaskedMimic's ProtoMotions)
    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4)

    Returns:
        The tangent and normal vectors in the quaternion representation. Shape is (..., 6)
    """
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = math_utils.quat_apply(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = math_utils.quat_apply(q, ref_norm)

    tan_norm = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return tan_norm


@torch.jit.script
def tan_norm_to_quat(tannorm: torch.Tensor) -> torch.Tensor:
    """Convert tangent and normal vectors to a quaternion. NOTE: assuming both tangent and normal vectors are normalized.
    Args:
        tannorm: The tangent and normal vectors in the quaternion representation. Shape is (..., 6)

    Returns:
        The quaternion in (x, y, z, w). Shape is (..., 4)
    """
    tan = tannorm[..., 0:3]
    norm = tannorm[..., 3:6]
    conj_axis = torch.cross(norm, tan, dim=len(tan.shape) - 1)
    matrix = torch.stack([tan, conj_axis, norm], dim=-1)
    quat = math_utils.quat_from_matrix(matrix)
    return quat


@torch.jit.script
def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Interpolate and sample across the quaternion (batch) based on the given t (batch).
    Args:
        q1, q2: The quaternion in (x, y, z, w). Shape is (B, 4)
        tau: The interpolation factor. Shape is (B,). Type is float where 0 <= t <= 1
    Return:
        The interpolated quaternion in (x, y, z, w). Shape is (B, 4)
    """
    # ensure the input is in the right shape
    assert q1.shape[-1] == 4, "The quaternion must be in (x, y, z, w) format."
    assert q2.shape[-1] == 4, "The quaternion must be in (x, y, z, w) format."
    assert tau.shape == q1.shape[:-1], "The batch size must be the same for all inputs."
    assert q1.shape[0] == q2.shape[0] == tau.shape[0], "The batch size must be the same for all inputs."
    assert (tau >= 0).all() and (tau <= 1).all(), "The interpolation factor must be in (0, 1) range."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True).clip(-1, 1)  # shape (B, 1)
    q2 = torch.where(dot_product < 0, -q2, q2)
    dot_product = torch.where(dot_product < 0, -dot_product, dot_product)

    # calculate the angle between the two quaternions
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)
    q_too_similar = (dot_product > (1 - 1e-9)) | (torch.abs(theta) < (1e-9))

    # avoid division by zero
    sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)

    # calculate the interpolation factor
    s1 = torch.sin((1 - tau).unsqueeze(-1) * theta) / sin_theta
    s2 = torch.sin(tau.unsqueeze(-1) * theta) / sin_theta

    # calculate the interpolated quaternion
    interpolated_quat = (s1 * q1 + s2 * q2) * (~q_too_similar) + q_too_similar * q1
    interpolated_quat = math_utils.normalize(interpolated_quat)

    return interpolated_quat


@torch.jit.script
def quat_angular_velocity(q_prev: torch.Tensor, q_next: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the angular velocity between two quaternions. (from q_prev to q_next in dt seconds)
    Args:
        q_prev, q_next: The quaternion in (x, y, z, w). Shape is (..., 4)
        dt: The time difference between the two quaternions.
    Returns:
        The angular velocity in the local frame. Shape is (..., 3)
    """
    # ensure the input is in the right shape
    assert q_prev.shape[-1] == 4, "The quaternion must be in (x, y, z, w) format."
    assert q_next.shape[-1] == 4, "The quaternion must be in (x, y, z, w) format."
    assert q_prev.shape == q_next.shape, "The shape of the two quaternions must be the same."
    assert dt > 0, "The time difference must be positive."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q_prev * q_next, dim=-1, keepdim=True).clip(-1, 1)  # shape (..., 1)
    q_next = torch.where(dot_product < 0, -q_next, q_next)

    quat_diff = math_utils.quat_mul(q_next, math_utils.quat_conjugate(q_prev))  # q_next * q_prev^-1
    axis_angle_diff = math_utils.axis_angle_from_quat(quat_diff)
    angular_velocity = axis_angle_diff / dt

    return angular_velocity
