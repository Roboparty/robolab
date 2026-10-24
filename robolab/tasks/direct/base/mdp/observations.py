# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.managers import SceneEntityCfg


def flatten_history(history, term_sizes, layout):
    if layout == "frame_major":
        return history.flatten(1)
    if layout == "term_major":
        return torch.cat([term.flatten(1) for term in history.split(term_sizes, dim=-1)], dim=-1)
    raise ValueError(f"Unsupported observation history layout: {layout!r}")


def feet_clearance(
    env, asset_cfg=SceneEntityCfg("robot"), sensor_cfgs=None,
    ankle_height=0.04, replace_invalid_hits=False,
):
    if sensor_cfgs:
        values = []
        for sensor_cfg in sensor_cfgs:
            sensor = env.scene[sensor_cfg.name]
            hits = sensor.data.ray_hits_w.torch[..., 2]
            if replace_invalid_hits:
                hits = torch.where(torch.isfinite(hits), hits, torch.zeros_like(hits))
            values.append(sensor.data.pos_w.torch[:, 2] - hits.mean(dim=-1))
        height = torch.stack(values, dim=-1)
    else:
        height = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    height = (height - ankle_height).clamp(0.0, 1.0)
    return torch.nan_to_num(height, nan=1.0, posinf=1.0, neginf=0.0)


def height_scan(env):
    pos_z = env.height_scanner.data.pos_w.torch[:, 2].unsqueeze(1)
    hits = env.height_scanner.data.ray_hits_w.torch[..., 2]
    offset = env.cfg.normalization.height_scan_offset
    invalid_value = env.cfg.normalization.height_scan_invalid_value
    if invalid_value is not None:
        hits = torch.where(torch.isfinite(hits), hits, pos_z - offset - invalid_value)
    height = (pos_z - hits - offset).clamp(-1.0, 1.0)
    return torch.nan_to_num(height, nan=1.0, posinf=1.0, neginf=-1.0) * env.obs_scales.height_scan
