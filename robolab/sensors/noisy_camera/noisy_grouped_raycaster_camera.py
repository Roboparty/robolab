from __future__ import annotations

from typing import TYPE_CHECKING

from ..grouped_ray_caster import GroupedRayCasterCamera
from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


class NoisyGroupedRayCasterCamera(NoisyCameraMixin, GroupedRayCasterCamera):
    cfg: NoisyGroupedRayCasterCameraCfg
