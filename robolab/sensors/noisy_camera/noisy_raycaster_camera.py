from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera
from isaaclab.utils.backend_utils import FactoryBase

from .noisy_camera import NoisyCameraMixin
from ..grouped_ray_caster.grouped_ray_caster import _backend_sensor_type

if TYPE_CHECKING:
    from .noisy_raycaster_camera_cfg import NoisyRayCasterCameraCfg


class NoisyRayCasterCamera(NoisyCameraMixin, BaseRayCasterCamera):
    cfg: NoisyRayCasterCameraCfg

    def __new__(cls, *args, **kwargs):
        return object.__new__(_backend_sensor_type(cls, FactoryBase._get_backend()))
