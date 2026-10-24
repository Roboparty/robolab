
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

import os

from robolab.assets.robots import RPO_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass
from robolab import ROBOLAB_ROOT_DIR

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
import robolab.tasks.manager_based.beyondmimic.mdp as mdp
from isaaclab.managers import SceneEntityCfg


@configclass
class RPOGetupMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_bm", "getup_supin2prone.npz"
        )
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            'left_thigh_yaw_link', 
            'right_thigh_yaw_link', 
            "base_link",
            'torso_link', 
            'left_thigh_roll_link', 
            'right_thigh_roll_link', 
            # 'left_arm_pitch_link', 
            # 'right_arm_pitch_link', 
            'left_thigh_pitch_link', 
            'right_thigh_pitch_link', 
            # 'left_arm_roll_link', 
            # 'right_arm_roll_link', 
            'left_knee_link', 
            'right_knee_link', 
            'left_arm_yaw_link', 
            'right_arm_yaw_link', 
            # 'left_ankle_pitch_link', 
            # 'right_ankle_pitch_link', 
            # 'left_elbow_pitch_link', 
            # 'right_elbow_pitch_link', 
            'left_ankle_roll_link', 
            'right_ankle_roll_link', 
            'left_elbow_yaw_link', 
            'right_elbow_yaw_link',
        ]

        # Resample the motion and robot state when the clip ends.
        self.commands.motion.reset_on_motion_end = True
        
        self.rewards.motion_body_pos.weight = 2.0
        self.rewards.stand_still_after_motion = RewTerm(
            func=mdp.stand_still_after_motion,
            weight=-0.2,
            params={
                "command_name": "motion",
                "pos_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "vel_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_weight": 1.0,
                "vel_weight": 0.04,
            },
        )

        self.events.randomize_push_robot.interval_range_s = (0.0, 5.0)

        self.episode_length_s = 5.0
