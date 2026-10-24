from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from isaaclab.utils.backend_utils import FactoryBase

from .noisy_camera import NoisyCameraMixin
from ..grouped_ray_caster.grouped_ray_caster import _backend_sensor_type

if TYPE_CHECKING:
    from .noisy_multi_mesh_ray_caster_camera_cfg import NoisyMultiMeshRayCasterCameraCfg


class NoisyMultiMeshRayCasterCamera(NoisyCameraMixin, BaseMultiMeshRayCasterCamera):
    cfg: NoisyMultiMeshRayCasterCameraCfg

    def __new__(cls, *args, **kwargs):
        return object.__new__(_backend_sensor_type(cls, FactoryBase._get_backend()))
