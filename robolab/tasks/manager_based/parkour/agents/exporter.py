# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export Parkour's full policy and separate encoder/latent actor graphs."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn

from rsl_rl.models.encoder_model import EncoderModel
from rsl_rl.runners import OnPolicyRunner


def export_policy(runner: OnPolicyRunner, path: str) -> None:
    """Keep standard exports and add the two graphs used by Parkour deployment."""
    runner.export_policy_to_jit(path=path, filename="policy.pt")
    runner.export_policy_to_onnx(path=path, filename="policy.onnx")
    policy = runner.alg.get_policy()
    for index, name in enumerate(policy.obs_groups_2d):
        encoder = copy.deepcopy(policy.cnns[name]).cpu().eval()
        dummy_input = torch.zeros(1, policy.obs_channels_2d[index], *policy.obs_dims_2d[index])
        torch.onnx.export(
            encoder,
            dummy_input,
            os.path.join(path, f"{index}-{name}_encoder.onnx"),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
        )
    actor = _LatentPolicyExporter(policy).cpu().eval()
    torch.onnx.export(
        actor,
        torch.zeros(1, policy._get_latent_dim()),
        os.path.join(path, "actor.onnx"),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"[INFO] Exported Parkour encoder and actor to: {path}")


class _LatentPolicyExporter(nn.Module):
    def __init__(self, model: EncoderModel) -> None:
        super().__init__()
        self.vector_dim = model.obs_dim
        self.normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        normalized = torch.cat(
            (self.normalizer(latent[..., : self.vector_dim]), latent[..., self.vector_dim :]), dim=-1
        )
        return self.output(self.mlp(normalized))
