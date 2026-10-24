# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics choices shared by RoboLab's Torch environments."""

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class PhysicsPresetCfg(PresetCfg):
    """Select PhysX or the Newton MJWarp solver with ``physics=NAME``."""

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            cone="pyramidal",
            impratio=1.0,
            update_data_interval=1,
            njmax=512,
            nconmax=256,
        ),
        num_substeps=1,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class RoughPhysicsPresetCfg(PhysicsPresetCfg):
    """Use Newton mesh contacts for procedural rough terrain."""

    # Match legacy RP1 Rough capacity; 8192 mesh environments overflow the 64 MiB default.
    isaacsim_physx: PhysxCfg = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
        gpu_collision_stack_size=2**29,
    )

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            cone="pyramidal",
            impratio=1.0,
            update_data_interval=2,
            njmax=1024,
            nconmax=512,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=5_000_000),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=2,
    )
    default: NewtonCfg = newton_mjwarp
