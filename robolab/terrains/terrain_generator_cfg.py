from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils import configclass

from .terrain_generator import FiledTerrainGenerator

@configclass
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator

    preserve_heightfield_samples: bool = True
    """Keep authored parkour surfaces when converting to triangle meshes."""

    one_col_per_subterrain: bool = True
    """Whether to use one column per sub-terrain type. Defaults to False.

    If True, ``num_cols`` is set to ``len(sub_terrains)`` (one type per column), and
    environment counts follow each sub-terrain's ``proportion``. If False, columns are
    mapped by cumulative proportion and environments are split uniformly (Isaac Lab default).
    """
