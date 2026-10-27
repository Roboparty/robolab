from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg

class FiledTerrainGenerator(TerrainGenerator):
    """Terrain generator that records per-cell sub-terrain configs and indices."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):
        # Per-cell sub-terrain index; filled during terrain generation (row, col) indexing.
        self.subterrain_index_grid: np.ndarray | None = None
        # Flat list: self._subterrain_specific_cfgs[row * num_cols + col]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        # One column per sub-terrain: num_cols follows the number of types.
        if cfg.one_col_per_subterrain:
            cfg.num_cols = len(cfg.sub_terrains)
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        """Record the specific config for each sub-terrain mesh generation."""
        if self.cfg.preserve_heightfield_samples and hasattr(cfg, "preserve_heightfield_samples"):
            cfg = cfg.copy()
            cfg.preserve_heightfield_samples = True
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        self._subterrain_specific_cfgs.append(cfg)
        return mesh, origin

    def _generate_random_terrains(self):
        """Add terrains with random sub-terrain type per grid cell."""
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = int(self.np_rng.choice(len(proportions), p=proportions))
            self.subterrain_index_grid[sub_row, sub_col] = sub_index
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains with sub-terrain type fixed per column and difficulty along rows."""
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # One column per type, or map columns by cumulative proportion (Isaac Lab default).
        if self.cfg.one_col_per_subterrain:
            sub_indices = np.arange(self.cfg.num_cols, dtype=np.int32)
        else:
            sub_indices = []
            for index in range(self.cfg.num_cols):
                sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                sub_index = int(sub_indices[sub_col])
                self.subterrain_index_grid[sub_row, sub_col] = sub_index
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def get_subterrain_indices(
        self, row_ids: torch.Tensor | int, col_ids: torch.Tensor | int, device: str | torch.device | None = None
    ) -> torch.Tensor:
        """Return sub-terrain dict indices for grid cells (row, col), aligned with terrain_levels/types."""
        if self.subterrain_index_grid is None:
            raise RuntimeError("subterrain_index_grid is not initialized. Terrain generation may have failed.")
        grid = torch.as_tensor(self.subterrain_index_grid, device=device, dtype=torch.long)
        return grid[row_ids, col_ids]

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all sub-terrains."""
        return self._subterrain_specific_cfgs.copy()

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a sub-terrain by its row and column index."""
        num_cols = self.cfg.num_cols
        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.cpu().numpy()
        if isinstance(col_ids, torch.Tensor):
            col_ids = col_ids.cpu().numpy()
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, np.ndarray):
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            return self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None
        return None
