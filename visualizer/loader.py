"""Load Gaussian Splatting data from trained PLY files."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from plyfile import PlyData

# DC spherical harmonic coefficient: RGB = sh_dc * C0 + 0.5
_SH_C0 = 0.28209479177387814

DEFAULT_PLY = (
    Path(__file__).parent.parent
    / "output/bear/point_cloud/iteration_1000/point_cloud.ply"
)


def load_gaussians(ply_path: str | Path = DEFAULT_PLY) -> dict:
    """Load Gaussian parameters from a trained PLY file.

    Returns a dict with:
        positions:   (N, 3) float32 – world-space Gaussian centres
        opacities:   (N,)   float32 – activated opacities in [0, 1]
        colors:      (N, 3) float32 – RGB in [0, 1] from DC SH coefficient
        covariances: (N, 3, 3) float32 – 3-D covariance matrices
        scales:      (N, 3) float32 – activated scales (exp applied)
        rotations:   (N, 4) float32 – unit quaternions [w, x, y, z]
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    opacities = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))).astype(
        np.float32
    )

    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(dc * _SH_C0 + 0.5, 0.0, 1.0)

    raw_scales = np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]], axis=1
    ).astype(np.float32)
    scales = np.exp(raw_scales)

    rotations = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1
    ).astype(np.float32)
    # Normalise quaternions – they should already be unit, but ensure it.
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / np.where(norms > 0, norms, 1.0)

    covariances = _compute_covariances(scales, rotations)

    return {
        "positions": positions,
        "opacities": opacities,
        "colors": colors,
        "covariances": covariances,
        "scales": scales,
        "rotations": rotations,
    }


def _compute_covariances(
    scales: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """Compute 3-D covariance matrices Σ = R S S^T R^T.

    Args:
        scales:    (N, 3) – activated (positive) scales per axis.
        rotations: (N, 4) – unit quaternions [w, x, y, z].

    Returns:
        (N, 3, 3) symmetric positive-definite covariance matrices.
    """
    R = _quaternions_to_matrices(rotations)  # (N, 3, 3)
    scales_sq = scales ** 2  # (N, 3)
    # R @ diag(scales_sq) = R * scales_sq  (broadcast over columns)
    RS = R * scales_sq[:, np.newaxis, :]  # (N, 3, 3)
    return (RS @ R.transpose(0, 2, 1)).astype(np.float32)  # (N, 3, 3)


def _quaternions_to_matrices(wxyz: np.ndarray) -> np.ndarray:
    """Convert N quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    w, x, y, z = wxyz[:, 0], wxyz[:, 1], wxyz[:, 2], wxyz[:, 3]
    N = len(wxyz)
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
