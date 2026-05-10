"""Surface Gaussian selection algorithm.

Selects the 'front layer' of Gaussians visible within a screen-space
rectangle from a given camera viewpoint.

Strategy: divide the selection rectangle into a coarse grid; within each
cell, depth-sort candidate Gaussians and accumulate opacity front-to-back
until the cumulative opacity reaches `opacity_threshold`.  Gaussians that
are reached before the threshold are considered part of the visible surface.
"""

from __future__ import annotations

import numpy as np

from .projector import project_gaussians

DEFAULT_OPACITY_THRESHOLD = 0.2
DEFAULT_GRID_SIZE = 24


def select_surface_gaussians(
    positions: np.ndarray,
    opacities: np.ndarray,
    camera_pos: np.ndarray,
    camera_wxyz: np.ndarray,
    fov_y: float,
    aspect: float,
    screen_rect: tuple[float, float, float, float],
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    """Return a boolean mask of surface Gaussians within a screen rectangle.

    Args:
        positions:         (N, 3) world-space Gaussian centres.
        opacities:         (N,)   activated opacities in [0, 1].
        camera_pos:        (3,)   camera world-space position.
        camera_wxyz:       (4,)   camera-to-world quaternion [w, x, y, z].
        fov_y:             Vertical FOV in radians.
        aspect:            Viewport width / height.
        screen_rect:       (min_x, min_y, max_x, max_y) in [0, 1] screen
                           coords (OpenCV: 0,0 = upper-left).
        opacity_threshold: Cumulative alpha at which a column is considered
                           opaque.  0.5 means include all Gaussians until
                           90 % of the incoming light is blocked.
        grid_size:         Resolution of the coarse grid used for per-column
                           opacity accumulation.

    Returns:
        selected: (N,) boolean array – True for surface Gaussians.
    """
    N = len(positions)
    screen_xy, depths, valid = project_gaussians(
        positions, camera_pos, camera_wxyz, fov_y, aspect
    )

    min_x, min_y, max_x, max_y = screen_rect
    in_rect = (
        valid
        & (screen_xy[:, 0] >= min_x)
        & (screen_xy[:, 0] <= max_x)
        & (screen_xy[:, 1] >= min_y)
        & (screen_xy[:, 1] <= max_y)
    )

    selected = np.zeros(N, dtype=bool)
    if not np.any(in_rect):
        return selected

    rect_indices = np.where(in_rect)[0]
    rect_screen = screen_xy[rect_indices]
    rect_depths = depths[rect_indices]
    rect_opacities = opacities[rect_indices]

    rect_w = max(max_x - min_x, 1e-6)
    rect_h = max(max_y - min_y, 1e-6)

    cell_x = (
        np.floor((rect_screen[:, 0] - min_x) / rect_w * grid_size)
        .astype(int)
        .clip(0, grid_size - 1)
    )
    cell_y = (
        np.floor((rect_screen[:, 1] - min_y) / rect_h * grid_size)
        .astype(int)
        .clip(0, grid_size - 1)
    )
    cell_id = cell_y * grid_size + cell_x

    selected_in_rect = np.zeros(len(rect_indices), dtype=bool)

    for cell in range(grid_size * grid_size):
        cell_mask = cell_id == cell
        if not np.any(cell_mask):
            continue

        cell_local_idx = np.where(cell_mask)[0]
        cell_depths = rect_depths[cell_mask]
        cell_opacities = rect_opacities[cell_mask]

        # Front-to-back order
        sort_order = np.argsort(cell_depths)

        transmittance = 1.0
        stop_threshold = 1.0 - opacity_threshold
        for rank_idx in sort_order:
            if transmittance <= stop_threshold:
                break
            selected_in_rect[cell_local_idx[rank_idx]] = True
            transmittance *= 1.0 - float(cell_opacities[rank_idx])

    selected[rect_indices[selected_in_rect]] = True
    return selected
