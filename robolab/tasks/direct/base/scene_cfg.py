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

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import preset
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.sensors.ray_caster import RayCasterCfg 

if TYPE_CHECKING:
    from .base_config import SceneContextCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration shared by the Direct locomotion tasks."""

    def __init__(self, config: "SceneContextCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.robot = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        self.left_feet_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(?!.*left_ankle_roll_link/).*left_ankle_roll_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, 20.0)),
            ray_alignment='yaw',
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            global_world_only=True,
            update_period=step_dt,
        )
        self.right_feet_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(?!.*right_ankle_roll_link/).*right_ankle_roll_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, 20.0)),
            ray_alignment='yaw',
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            global_world_only=True,
            update_period=step_dt,
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/(?!.*" + config.height_scanner.prim_body_name
                    + "/).*" + config.height_scanner.prim_body_name
                ),
                offset=RayCasterCfg.OffsetCfg(pos=config.height_scanner.offset),
                ray_alignment='yaw',
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                global_world_only=True,
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )


def configure_self_contact_sensors(scene, sensor_name, physics_dt, body_names):
    """Use native Newton filtering and per-body PhysX ground filters."""
    sensor = getattr(scene, sensor_name)
    # GA preserves terrain prim paths; the alternative covers older Newton mesh labels.
    terrain_shapes = ["/World/ground/.*|shape_[0-9]+"] if scene.terrain.terrain_type == "generator" else ["/World/ground/.*"]
    sensor.filter_shape_prim_expr = preset(
        default=terrain_shapes, isaacsim_physx=[], newton_mjwarp=terrain_shapes
    )
    ground_path = (
        "/World/ground/terrain/mesh" if scene.terrain.terrain_type == "generator"
        else "/World/ground/terrain/GroundPlane/CollisionPlane"
    )
    # PhysX supports only one sensor body per filtered contact sensor.
    for body in body_names:
        setattr(scene, "self_contact_" + body, preset(
            default=None,
            isaacsim_physx=ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*" + body,
                filter_prim_paths_expr=[ground_path],
                history_length=3,
                update_period=physics_dt,
            ),
            newton_mjwarp=None,
        ))
