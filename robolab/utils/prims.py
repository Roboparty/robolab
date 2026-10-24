""" Code snippet from accessing prims in isaac sim (scene) """

import omni.physics.tensors.api as physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils


def get_articulation_view(
    prim_path: str,
    physics_sim_view: physx.SimulationView,
) -> physx.ArticulationView:
    """create simulation view and the prim view (partially copied from `ray_caster.py`)"""
    found_supported_prim_class = False
    prim = sim_utils.find_first_matching_prim(prim_path)
    if prim is None:
        raise RuntimeError(f"Failed to find a prim at path expression: {prim_path}")
    # create view based on the type of prim
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulation_view: physx.ArticulationView = physics_sim_view.create_articulation_view(
            sim_utils.path_expr_to_glob(prim_path)
        )
        found_supported_prim_class = True
    if not found_supported_prim_class:
        raise RuntimeError(
            f"Failed to find a valid prim view class for the prim paths: {prim_path}, For robot motion reference, only"
            " accept articulated prims for now."
        )

    return articulation_view
