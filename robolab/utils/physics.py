# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics choices shared by RoboLab's Torch environments."""

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class PhysicsPresetCfg(PresetCfg):
    """Select PhysX or the Newton MJWarp solver with ``physics=NAME``.

    ``njmax`` and ``nconmax`` are allocated per MuJoCo world.  The values
    below cover the flat RPO/RP1 and BeyondMimic workloads without reserving
    the much larger legacy rough-terrain budget for every environment.
    """

    isaacsim_physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            cone="pyramidal",
            impratio=1.0,
            update_data_interval=1,
            # RP1 self-contact can reach about 200 constraints in one world;
            # 256 leaves headroom while remaining below the old 512 budget.
            njmax=256,
            nconmax=32,
        ),
        num_substeps=1,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class RoughPhysicsPresetCfg(PhysicsPresetCfg):
    """Use Newton mesh contacts for procedural rough terrain."""

    # Rough mesh contacts need more headroom than a plane.  RP1 rough
    # self-contact reaches about 400 constraints in one world, so 512 keeps
    # a margin while halving the legacy 1024/512 budget.
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
            njmax=512,
            nconmax=64,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=5_000_000),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=2,
    )
    default: NewtonCfg = newton_mjwarp
