# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class ArmClearanceCfg:
    """Roll-dependent arm clearance, measured in each robot's canonical joint axes."""

    roll_clearance: float = MISSING
    backward_pitch_limit: float = MISSING
    pitch_per_roll: float = MISSING
    forward_yaw_margin: float = MISSING
    outward_yaw_per_roll: float = MISSING
    low_roll_elbow_flexion: float = MISSING
    elbow_flexion_per_roll: float = MISSING
    reverse_elbow_roll_start: float = MISSING
    reverse_elbow_range: float = MISSING
    reverse_elbow_roll_span: float = MISSING
    overhead_roll_start: float = MISSING
    overhead_yaw_slope: float = MISSING
    overhead_yaw_margin: float = MISSING
    reverse_elbow_yaw_scale: float = MISSING


@configclass
class InterruptCfg:
    use_interrupt: bool = False
    max_curriculum: float = 1.0
    curriculum_pass_distance: float | None = None
    interrupt_ratio: float = 0.5
    interrupt_joint_names: list = []
    interrupt_scale: list = []
    interrupt_lower_bound: list = []
    interrupt_init_range: float = 0.2
    interrupt_update_step: int = 30
    switch_prob: float = 0.005
    arm_joint_signs: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0)
    low_roll_threshold: float = 0.5
    low_roll_yaw_range: tuple = (-1.57, 0.85)
    arm_clearance: ArmClearanceCfg | None = None
