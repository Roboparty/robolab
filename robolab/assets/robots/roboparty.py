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


from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg
from isaaclab_tasks.utils import preset

from robolab.assets import ISAAC_DATA_DIR
from robolab.assets.robots.rp1_actuators import RP1ActuatorCfg

# Keep legacy policy and motion-file coordinates identical across solvers.
RPO_JOINT_ORDER = (
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint",
    "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint",
    "left_arm_roll_joint", "right_arm_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_arm_yaw_joint", "right_arm_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)
RPO_BODY_ORDER = ("base_link", *(name.removesuffix("_joint") + "_link" for name in RPO_JOINT_ORDER))

RPO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    joint_ordering=RPO_JOINT_ORDER,
    body_ordering=RPO_BODY_ORDER,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            joint_effort_limit=120.0,
            joint_velocity_limit=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            joint_effort_limit=27.0,
            joint_velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            joint_effort_limit=27.0,
            joint_velocity_limit=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            joint_effort_limit=27.0,
            joint_velocity_limit=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]


RP1_ASSET_DIR = Path(ISAAC_DATA_DIR).resolve() / "robots/roboparty/rp1"
RP1_USD_DIR = Path(ISAAC_DATA_DIR).resolve() / "generated/rp1"
RP1_MJCF_PATH = str(RP1_ASSET_DIR / "mjcf/robot_24dof.xml")

# These coordinates match the published RP1 motion archives on either solver.
RP1_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_joint",
)
RP1_BODY_ORDER = ("base_link", *(name.removesuffix("_joint") + "_link" for name in RP1_JOINT_ORDER))
RP1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
RP1_BASE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
RP1_LEFT_FOOT_PRIM_PATH = (
    RP1_BASE_PRIM_PATH + "/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/"
    "left_knee_link/left_ankle_pitch_link/left_ankle_roll_link"
)
RP1_RIGHT_FOOT_PRIM_PATH = (
    RP1_BASE_PRIM_PATH + "/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/"
    "right_knee_link/right_ankle_pitch_link/right_ankle_roll_link"
)
RP1_DEFAULT_JOINT_POS = dict.fromkeys(RP1_JOINT_ORDER, 0.0)
RP1_DEFAULT_JOINT_POS.update({
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": -0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": -1.2,
    "right_hip_pitch_joint": 0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_shoulder_pitch_joint": -0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.2,
})

RP1_EFFORT_LIMITS = {
    ".*(waist|hip|knee).*": 145.53,
    ".*ankle.*": 56.0,
    ".*(shoulder|elbow|wrist).*": 28.0,
}
RP1_VELOCITY_LIMITS = {
    ".*(waist|hip|knee).*": 13.61357,
    ".*(ankle|shoulder|elbow|wrist).*": 20.94395,
}
RP1_RATED_EFFORT = {
    ".*(waist|hip|knee).*": 44.0,
    ".*ankle.*": 19.0,
    ".*(shoulder|elbow|wrist).*": 9.5,
}
RP1_RATED_VELOCITY = {
    ".*(waist|hip|knee).*": 8.37758,
    ".*(ankle|shoulder|elbow|wrist).*": 6.28319,
}

RP1_24DOF_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=RP1_MJCF_PATH,
        usd_dir=preset(
            default=str(RP1_USD_DIR / "mujoco"),
            isaacsim_physx=str(RP1_USD_DIR / "physx"),
            newton_mjwarp=str(RP1_USD_DIR / "mujoco"),
        ),
        physics_variant=preset(
            default="mujoco",
            isaacsim_physx="physx",
            newton_mjwarp="mujoco",
        ),
        self_collision=True,
        fix_base=False,
        activate_contact_sensors=True,
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos=RP1_DEFAULT_JOINT_POS.copy(),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    joint_ordering=RP1_JOINT_ORDER,
    body_ordering=RP1_BODY_ORDER,
    actuators={
        "all_joints": RP1ActuatorCfg(
            joint_names_expr=[".*"],
            actuator_effort_limit=RP1_EFFORT_LIMITS.copy(),
            actuator_velocity_limit=RP1_VELOCITY_LIMITS.copy(),
            joint_effort_limit=RP1_EFFORT_LIMITS.copy(),
            joint_velocity_limit=RP1_VELOCITY_LIMITS.copy(),
            keep_velocity={
                ".*(waist|hip|knee).*": 8.37758,
                ".*(ankle|shoulder|elbow|wrist).*": 15.70796,
            },
            rated_effort=RP1_RATED_EFFORT.copy(),
            rated_velocity=RP1_RATED_VELOCITY.copy(),
            stiffness={
                ".*waist_roll.*": 150.0,
                ".*waist_yaw.*": 150.0,
                ".*_hip_pitch.*": 120.0,
                ".*_hip_roll.*": 120.0,
                ".*_hip_yaw.*": 90.0,
                ".*_knee_.*": 120.0,
                ".*_ankle_(pitch|roll).*": 40.0,
                ".*_shoulder_pitch_joint": 30.0,
                ".*_shoulder_roll_joint": 30.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*elbow.*": 30.0,
                ".*_wrist_joint": 20.0,
            },
            damping={
                ".*waist_roll.*": 12.5,
                ".*waist_yaw.*": 12.5,
                ".*_hip_pitch.*": 9.0,
                ".*_hip_roll.*": 9.0,
                ".*_hip_yaw.*": 6.0,
                ".*_knee_.*": 9.0,
                ".*_ankle_(pitch|roll).*": 2.0,
                ".*_shoulder_pitch_joint": 2.5,
                ".*_shoulder_roll_joint": 2.5,
                ".*_shoulder_yaw_joint": 1.5,
                ".*elbow.*": 2.5,
                ".*_wrist_joint": 1.5,
            },
            armature={
                ".*(waist|hip|knee).*": 0.020000,
                ".*(ankle|shoulder|elbow|wrist).*": 0.010000,
            },
            min_delay=0,
            max_delay=2,
            friction=0.1,
            dynamic_friction=0.1,
            viscous_friction={
                ".*(waist|hip|knee).*": 0.02,
                ".*(ankle|shoulder|elbow|wrist).*": 0.01,
            },
        ),
    },
)
