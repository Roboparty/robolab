import copy
import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RPO_CFG, RPO_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import KEY_BODY_NAMES, ROUGH_TERRAINS_CFG, ParkourEnvCfg

RPO_CFG.init_state.pos = (0.0, 0.0, 0.85)
AMP_NUM_STEPS = 3


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class RPOParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(RPO_LINKS))
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_amp"
        )
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1,
            "36_11": 1,
            "114_08": 1,
            "114_09": 1,
            "A1-_Stand_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "turn_l": 1,
            "turn_r": 1,
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058



@configclass
class RPOParkourRoughEnvCfg_PLAY(RPOParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        # self.commands.base_velocity.velocity_ranges["pyramid_stairs"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["pyramid_stairs_inv_high_ground_aligned"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        # self.commands.base_velocity.velocity_ranges["hf_pyramid_slope_inv"] = {"lin_vel_x": (1.0, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.scene.leg_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class RPOParkourEnvCfg(RPOParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()


@configclass
class RPOParkourEnvCfg_PLAY(RPOParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_shoe_config()
