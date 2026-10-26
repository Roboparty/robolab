import copy
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RPO_CFG, RPO_LINKS
from robolab.sensors import Grid3dPointsGeneratorCfg, get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
]

RPO_CFG.init_state.pos = (0.0, 0.0, 0.85)
AMP_NUM_STEPS = 3

FEET_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.08,
    x_max=0.11,
    x_num=20,
    y_min=-0.025,
    y_max=0.025,
    y_num=6,
    z_min=-0.05,
    z_max=-0.02,
    z_num=4,
)

FOOT_HEIGHT_SCAN_RESOLUTION = 0.01
FOOT_HEIGHT_SCAN_RAY_START_HEIGHT = 20.0
FOOT_HEIGHT_SCAN_CENTER = (
    0.5 * (FEET_VOLUME_POINTS_GRID.x_min + FEET_VOLUME_POINTS_GRID.x_max),
    0.5 * (FEET_VOLUME_POINTS_GRID.y_min + FEET_VOLUME_POINTS_GRID.y_max),
)
FOOT_HEIGHT_SCAN_SIZE = (
    FEET_VOLUME_POINTS_GRID.x_max - FEET_VOLUME_POINTS_GRID.x_min,
    0.04,
)

KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.03,
    x_max=0.04,
    x_num=8,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.3,
    z_max=0.0,
    z_num=31,
)

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class RPOParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator = FEET_VOLUME_POINTS_GRID
        self.scene.knee_volume_points.points_generator = KNEE_VOLUME_POINTS_GRID
        for scanner_cfg in (
            self.scene.left_height_scanner,
            self.scene.right_height_scanner,
        ):
            scanner_cfg.offset.pos = (
                *FOOT_HEIGHT_SCAN_CENTER,
                FOOT_HEIGHT_SCAN_RAY_START_HEIGHT,
            )
            scanner_cfg.pattern_cfg.size = FOOT_HEIGHT_SCAN_SIZE
            scanner_cfg.pattern_cfg.resolution = FOOT_HEIGHT_SCAN_RESOLUTION
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/(?!.*torso_link[/]).*torso_link"
        self.scene.camera.offset.pos = (0.0875, 0.01, 0.20568)
        self.scene.camera.offset.rot = (0.0, 0.5, 0.0, 0.866)
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(RPO_LINKS))
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_amp"
        )
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.target_fps = 120
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
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }
        self.rewards.rewards.leg_yaw_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_thigh_yaw_joint"],
        )
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="torso_link"
        )
        self.rewards.rewards.pelvis_ang_vel_xy_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="torso_link"
        )
        self.rewards.rewards.joint_deviation_upper_body.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_arm_.*_joint", ".*_elbow_.*_joint"]
        )
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="torso_link"
        )
        if hasattr(self.events, "add_base_mass") and self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names="torso_link"
            )
        if hasattr(self.events, "randomize_rigid_body_com") and self.events.randomize_rigid_body_com is not None:
            self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names=["torso_link", "base_link"]
            )



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
