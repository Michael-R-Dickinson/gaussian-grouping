"""Camera projection utilities for mapping 3-D Gaussian centres to screen space.

Viser camera conventions (OpenCV):
    - wxyz quaternion describes camera-to-world rotation R_c2w.
    - Camera +X = right, +Y = down, +Z = forward (into scene).
    - Screen coordinates: (0, 0) = upper-left, (1, 1) = lower-right.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Return the (3, 3) camera-to-world rotation matrix for a single wxyz quaternion."""
    w, x, y, z = wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def project_gaussians(
    positions: np.ndarray,
    camera_pos: np.ndarray,
    camera_wxyz: np.ndarray,
    fov_y: float,
    aspect: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3-D Gaussian centres to normalised screen space.

    Args:
        positions:   (N, 3) world-space positions.
        camera_pos:  (3,)   camera position in world space.
        camera_wxyz: (4,)   camera-to-world rotation as [w, x, y, z].
        fov_y:       Vertical field of view in radians.
        aspect:      Viewport width / height.

    Returns:
        screen_xy: (N, 2) normalised screen coords in [0, 1]. x=left→right,
                   y=top→bottom (OpenCV image convention).
        depths:    (N,)   depth (distance along camera +Z axis). Positive
                   values indicate the point is in front of the camera.
        valid:     (N,)   bool mask – True when depth > 0 (point is visible).
    """
    R_c2w = wxyz_to_rotation_matrix(camera_wxyz)  # (3, 3)
    R_w2c = R_c2w.T

    p_rel = positions - camera_pos[np.newaxis, :]  # (N, 3)
    p_cam = (R_w2c @ p_rel.T).T  # (N, 3) in camera space

    depths = p_cam[:, 2].astype(np.float64)
    valid = depths > 0.0

    tan_half_fov_y = np.tan(fov_y / 2.0)
    tan_half_fov_x = tan_half_fov_y * aspect

    z_safe = np.where(valid, depths, 1.0)

    ndc_x = p_cam[:, 0] / (z_safe * tan_half_fov_x)   # [-1, 1]: neg=left
    ndc_y = p_cam[:, 1] / (z_safe * tan_half_fov_y)   # [-1, 1]: neg=top (y-down)

    screen_x = (ndc_x + 1.0) / 2.0   # [0, 1]: 0=left
    screen_y = (ndc_y + 1.0) / 2.0   # [0, 1]: 0=top

    screen_xy = np.stack([screen_x, screen_y], axis=1).astype(np.float32)
    return screen_xy, depths.astype(np.float32), valid


def camera_look_at_to_wxyz(
    camera_pos: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Compute the viser wxyz quaternion for a camera position + look-at point.

    Replicates viser's internal _update_wxyz logic, which follows OpenCV
    convention: camera +Z = forward, camera +Y = down (image), camera +X
    = image right.

    The `up` parameter is the world-space direction that should appear at the
    TOP of the rendered image (e.g. world +Y for a +Y-up scene).

    Returns:
        wxyz: (4,) quaternion [w, x, y, z].
    """
    z = np.array(look_at, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
    z /= np.linalg.norm(z)

    # Viser rotates the world-up vector by 180° around the camera forward axis
    # to obtain the camera's +Y direction (which points DOWN in image space).
    R_flip = 2.0 * np.outer(z, z) - np.eye(3)
    y = R_flip @ np.array(up, dtype=np.float64)
    y -= np.dot(z, y) * z  # project onto plane perpendicular to z
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(up[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        y = R_flip @ alt
        y -= np.dot(z, y) * z
        y /= np.linalg.norm(y)
    else:
        y /= y_norm

    x = np.cross(y, z)  # camera +X = image right
    R_c2w = np.stack([x, y, z], axis=1)  # columns = camera axes in world space
    q = Rotation.from_matrix(R_c2w).as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # [w, x, y, z]
