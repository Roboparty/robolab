# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Name-based symmetry for RP1 Direct observations and actions."""

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def joint_mirror_spec(joint_names):
    indices_by_name = {name: index for index, name in enumerate(joint_names)}
    if len(indices_by_name) != 24 or len(joint_names) != 24:
        raise ValueError("RP1 symmetry requires 24 unique joint names.")
    indices, signs = [], []
    for name in joint_names:
        if name.startswith("left_"):
            other = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            other = "left_" + name.removeprefix("right_")
        elif name in ("waist_roll_joint", "waist_yaw_joint"):
            other = name
        else:
            raise ValueError(f"Unsupported RP1 joint: {name!r}.")
        if other not in indices_by_name:
            raise ValueError(f"Missing RP1 mirrored joint: {other!r}.")
        indices.append(indices_by_name[other])
        # RP1's ankle-pitch axes share a sign; the remaining axes reverse.
        signs.append(1.0 if name.endswith("ankle_pitch_joint") else -1.0)
    return tuple(indices), tuple(signs)


def mirror_joints(values, joint_names):
    if values.shape[-1] != 24:
        raise ValueError(f"RP1 joint tensor has {values.shape[-1]} columns, expected 24.")
    indices, signs = joint_mirror_spec(tuple(joint_names))
    return values[..., list(indices)] * values.new_tensor(signs)


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _feet_indices(env):
    sensor = env.contact_sensor
    body_ids = tuple(env.feet_cfg.body_ids)
    cached = getattr(env, "_rp1_symmetry_feet_spec", None)
    if cached is not None and cached[0] is sensor and cached[1] == body_ids:
        return cached[2]
    # PhysX resolves paths for every environment when body_names is accessed.
    body_names = sensor.body_names
    names = tuple(body_names[index] for index in body_ids)
    if set(names) != {"left_ankle_roll_link", "right_ankle_roll_link"} or len(names) != 2:
        raise ValueError(f"RP1 symmetry requires exactly two ankle-roll feet, got {names}.")
    indices = (1, 0)
    env._rp1_symmetry_feet_spec = (sensor, body_ids, indices)
    return indices


def _mirror_height_scan(env, values):
    pattern = env.cfg.scene.height_scanner.pattern_cfg
    x_count = int(round(pattern.size[0] / pattern.resolution)) + 1
    y_count = int(round(pattern.size[1] / pattern.resolution)) + 1
    if values.shape[-1] != x_count * y_count:
        raise ValueError("RP1 height-scan width does not match the configured grid.")
    if pattern.ordering == "xy":
        return values.reshape(*values.shape[:-1], y_count, x_count).flip(-2).reshape(values.shape)
    if pattern.ordering == "yx":
        return values.reshape(*values.shape[:-1], x_count, y_count).flip(-1).reshape(values.shape)
    raise ValueError(f"Unsupported height grid ordering: {pattern.ordering!r}.")


def mirror_observation(env, values, *, critic=False):
    env = _base_env(env)
    joint_names = tuple(env.robot.joint_names)
    has_interrupt = getattr(getattr(env.cfg, "interrupt", None), "use_interrupt", False)
    actor_width = 82 if has_interrupt else 81
    frame_width = actor_width + 63 if critic else actor_width
    has_scan = env.cfg.scene_context.height_scanner.enable_height_scan
    if has_scan and (critic or env.cfg.scene_context.height_scanner.enable_height_scan_actor):
        pattern = env.cfg.scene.height_scanner.pattern_cfg
        frame_width += (int(round(pattern.size[0] / pattern.resolution)) + 1) * (
            int(round(pattern.size[1] / pattern.resolution)) + 1
        )
    history_length = env.cfg.robot.critic_obs_history_length if critic else env.cfg.robot.actor_obs_history_length
    if values.shape[-1] != frame_width * history_length:
        raise ValueError(
            f"RP1 observation has {values.shape[-1]} columns, expected {frame_width * history_length}."
        )
    layout = getattr(env.cfg.robot, "observation_history_layout", "frame_major")
    if layout not in ("frame_major", "term_major"):
        raise ValueError(f"Unsupported RP1 observation history layout: {layout!r}.")
    term_widths = [3, 3, 3, 24, 24, 24]
    if has_interrupt:
        term_widths.append(1)
    if critic:
        term_widths.extend((3, 2, 6, 2, 2, 24, 24))
    if has_scan and (critic or env.cfg.scene_context.height_scanner.enable_height_scan_actor):
        term_widths.append(frame_width - sum(term_widths))
    if layout == "term_major":
        offset, terms = 0, []
        for width in term_widths:
            end = offset + history_length * width
            terms.append(values[..., offset:end].reshape(*values.shape[:-1], history_length, width))
            offset = end
        frames = torch.cat(terms, dim=-1)
    else:
        frames = values.reshape(*values.shape[:-1], history_length, frame_width)
    mirrored = frames.clone()
    mirrored[..., :9] *= values.new_tensor((-1, 1, -1, 1, -1, 1, 1, -1, -1))
    for offset in (9, 33, 57):
        mirrored[..., offset:offset + 24] = mirror_joints(frames[..., offset:offset + 24], joint_names)
    offset = actor_width
    if critic:
        feet = _feet_indices(env)
        mirrored[..., offset:offset + 3] *= values.new_tensor((1, -1, 1))
        offset += 3
        mirrored[..., offset:offset + 2] = frames[..., offset:offset + 2][..., list(feet)]
        offset += 2
        forces = frames[..., offset:offset + 6].reshape(*frames.shape[:-1], 2, 3)
        mirrored[..., offset:offset + 6] = (
            forces[..., list(feet), :] * values.new_tensor((1, -1, 1))
        ).reshape(*frames.shape[:-1], 6)
        offset += 6
        for _ in range(2):
            mirrored[..., offset:offset + 2] = frames[..., offset:offset + 2][..., list(feet)]
            offset += 2
        for _ in range(2):
            mirrored[..., offset:offset + 24] = mirror_joints(frames[..., offset:offset + 24], joint_names)
            offset += 24
    if has_scan and (critic or env.cfg.scene_context.height_scanner.enable_height_scan_actor):
        mirrored[..., offset:] = _mirror_height_scan(env, frames[..., offset:])
    if layout == "term_major":
        offset, terms = 0, []
        for width in term_widths:
            terms.append(mirrored[..., offset:offset + width].reshape(*values.shape[:-1], history_length * width))
            offset += width
        return torch.cat(terms, dim=-1)
    return mirrored.reshape(values.shape)


@torch.no_grad()
def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        mirrored = obs.clone()
        if "policy" in obs.keys():
            mirrored["policy"] = mirror_observation(env, obs["policy"])
        if "critic" in obs.keys():
            mirrored["critic"] = mirror_observation(env, obs["critic"], critic=True)
        obs_aug = torch.cat((obs, mirrored), dim=0)
    if actions is None:
        actions_aug = None
    else:
        env = _base_env(env)
        actions_aug = torch.cat((actions, mirror_joints(actions, tuple(env.robot.joint_names))), dim=0)
    return obs_aug, actions_aug
