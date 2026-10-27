"""Authored height-field surfaces for parkour (sharp stairs, no slope blend)."""

import numpy as np
import trimesh


def heightfield_mesh(heights, spacing, offset=(0.0, 0.0, 0.0)):
    """Build an unshifted grid mesh with vertical perimeter sides."""
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

    heights = np.asarray(heights, dtype=np.float32)
    if heights.ndim != 2 or min(heights.shape) < 2 or not np.isfinite(heights).all():
        raise ValueError("Terrain requires a finite height array of at least 2x2.")
    sx, sy = np.broadcast_to(spacing, (2,))
    if min(sx, sy) <= 0:
        raise ValueError("Terrain grid spacing must be positive.")
    vertices, faces = convert_height_field_to_mesh(heights, 1.0, 1.0, None)
    vertices[:, :2] *= (sx, sy)
    nx, ny = heights.shape
    perimeter = np.r_[
        np.arange(ny),
        np.arange(1, nx) * ny + ny - 1,
        (nx - 1) * ny + np.arange(ny - 2, -1, -1),
        np.arange(nx - 2, 0, -1) * ny,
    ]
    lower = vertices[perimeter].copy()
    lower[:, 2] = heights.min() - 1e-4
    bottom = np.arange(len(lower)) + len(vertices)
    next_top, next_bottom = np.roll(perimeter, -1), np.roll(bottom, -1)
    sides = np.concatenate(
        (
            np.stack((perimeter, next_top, bottom), axis=1),
            np.stack((next_top, next_bottom, bottom), axis=1),
        )
    )
    vertices = np.concatenate((vertices, lower))
    faces = np.concatenate((faces, sides))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.apply_translation(offset)
    return mesh


def terrace_meshes(heights, levels, spacing):
    """Partition stair levels into boxes with optional rough heightfield caps."""
    heights = np.asarray(heights, dtype=np.float64)
    levels = np.asarray(levels)
    if heights.shape != levels.shape:
        raise ValueError("Stair level and height arrays must have the same shape.")
    nx, ny = heights.shape
    rectangles, active = [], {}
    for x in range(nx):
        cuts = np.r_[0, np.flatnonzero(np.diff(levels[x])) + 1, ny]
        following = {}
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            key = (int(y0), int(y1), float(levels[x, y0]))
            rect = active.pop(key, [x, x, int(y0), int(y1)])
            rect[1] = x + 1
            following[key] = rect
        rectangles.extend(active.values())
        active = following
    rectangles.extend(active.values())
    bottom = float(heights.min()) - 0.1
    meshes = []
    for x0, x1, y0, y1 in rectangles:
        lo = np.maximum((x0 - 0.5, y0 - 0.5), 0) * spacing
        hi = np.minimum((x1 - 0.5, y1 - 0.5), (nx - 1, ny - 1)) * spacing
        patch = heights[x0:x1, y0:y1]
        top = float(patch.min())
        box = trimesh.creation.box(
            extents=(*(hi - lo), top - bottom),
            transform=trimesh.transformations.translation_matrix(
                (*((lo + hi) / 2), (top + bottom) / 2)
            ),
        )
        meshes.append(box)
        if np.ptp(patch) > 0:
            count = np.maximum(np.rint((hi - lo) / (spacing / 2)).astype(int) + 1, 2)
            xs, ys = (np.linspace(lo[i], hi[i], count[i]) / spacing for i in range(2))
            along_y = np.stack([np.interp(ys, np.arange(y0, y1), row) for row in patch])
            cap = np.stack([np.interp(xs, np.arange(x0, x1), col) for col in along_y.T], axis=1)
            meshes.append(heightfield_mesh(cap, (hi - lo) / (count - 1), (*lo, 0.0)))
    return meshes
