from dataclasses import MISSING
import re

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass



@configclass
class GroupedRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type | str = "{DIR}.grouped_ray_caster:GroupedRayCaster"

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""


def get_link_prim_targets(
    links: list[str],
    prefix: str = "/World/envs/env_.*/Robot/",
    suffix: str | None = None,
    is_shared=True,  # whether the target prim is assumed to be the same mesh across all environments.
    **kwargs: dict,
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Target each link's visuals in both flat USD and hierarchical URDF-converter layouts.

    A custom suffix keeps the caller's path convention. The default targets either the
    legacy ``visuals`` container or the converted visual named after its link.
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=(
                f"{prefix}{link}{suffix}" if suffix is not None
                else (
                    f"{prefix}(?!.*{re.escape(link)}[/]{re.escape(link)}[/])"
                    f".*{re.escape(link)}[/](?:visuals|{re.escape(link)})"
                )
            ),
            is_shared=is_shared,
            **kwargs,
        )
        for link in links
    ]
