"""Configuration for the grouped-ray-cast camera sensor."""

from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .grouped_ray_caster_cfg import GroupedRayCasterCfg


@configclass
class GroupedRayCasterCameraCfg(GroupedRayCasterCfg):
    """Configuration for the grouped-ray-cast camera sensor."""

    class_type: type | str = "{DIR}.grouped_ray_caster_camera:GroupedRayCasterCamera"

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to identity."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

    data_types: list[str] = ["distance_to_image_plane"]
    """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf`` for ``distance_to_camera`` and ``nan``
      for ``distance_to_image_plane`` data type.
    """

    pattern_cfg: PinholeCameraPatternCfg = MISSING
    """The pattern that defines the local ray starting positions and directions in a pinhole camera pattern."""

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GroupedRayCasterCamera",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )

    def __post_init__(self):
        # for cameras, this quantity should be False always.
        self.ray_alignment = "base"
