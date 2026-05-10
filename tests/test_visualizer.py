"""Unit tests for the Gaussian Splatting visualizer.

Tests use the bear dataset (output/bear/point_cloud/iteration_1000/point_cloud.ply).
Run with:
    python -m pytest tests/test_visualizer.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from pathlib import Path

BEAR_PLY = (
    Path(__file__).parent.parent
    / "output/bear/point_cloud/iteration_1000/point_cloud.ply"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_matrix_z(theta: float) -> np.ndarray:
    """3×3 rotation about world Z axis by theta radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation matrix → viser wxyz quaternion."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]


# ---------------------------------------------------------------------------
# loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    """Tests for visualizer.loader.load_gaussians."""

    @pytest.fixture(scope="class")
    def data(self):
        from visualizer.loader import load_gaussians
        return load_gaussians(BEAR_PLY)

    def test_ply_exists(self):
        assert BEAR_PLY.exists(), f"Bear PLY not found at {BEAR_PLY}"

    def test_keys(self, data):
        for key in ("positions", "opacities", "colors", "covariances", "scales", "rotations"):
            assert key in data, f"Missing key: {key}"

    def test_shapes(self, data):
        N = data["positions"].shape[0]
        assert N > 100_000, f"Expected >100k Gaussians, got {N}"
        assert data["positions"].shape == (N, 3)
        assert data["opacities"].shape == (N,)
        assert data["colors"].shape == (N, 3)
        assert data["covariances"].shape == (N, 3, 3)
        assert data["scales"].shape == (N, 3)
        assert data["rotations"].shape == (N, 4)

    def test_opacity_range(self, data):
        assert data["opacities"].min() >= 0.0
        assert data["opacities"].max() <= 1.0

    def test_color_range(self, data):
        assert data["colors"].min() >= 0.0
        assert data["colors"].max() <= 1.0

    def test_scales_positive(self, data):
        assert (data["scales"] > 0).all(), "All scales must be positive after exp()"

    def test_rotation_unit_quaternions(self, data):
        norms = np.linalg.norm(data["rotations"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_covariances_symmetric(self, data):
        cov = data["covariances"]
        diff = np.abs(cov - cov.transpose(0, 2, 1))
        # Float32 arithmetic introduces small asymmetries; 1e-4 tolerance is generous.
        assert diff.max() < 1e-4, "Covariance matrices must be symmetric"

    def test_covariances_positive_semidefinite(self, data):
        sample = data["covariances"][::1000]
        eigvals = np.linalg.eigvalsh(sample)
        assert eigvals.min() >= -1e-4, "Covariance matrices must be PSD"


class TestSHConversion:
    """Tests for DC spherical-harmonic → RGB conversion."""

    def test_dc_zero_gives_half(self):
        from visualizer.loader import _SH_C0
        # f_dc = 0 → rgb = 0 * C0 + 0.5 = 0.5
        dc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        rgb = np.clip(dc * _SH_C0 + 0.5, 0, 1)
        np.testing.assert_allclose(rgb, [[0.5, 0.5, 0.5]], atol=1e-6)

    def test_dc_positive_saturates_to_one(self):
        from visualizer.loader import _SH_C0
        large = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        rgb = np.clip(large * _SH_C0 + 0.5, 0, 1)
        np.testing.assert_allclose(rgb, [[1.0, 1.0, 1.0]], atol=1e-6)

    def test_dc_negative_saturates_to_zero(self):
        from visualizer.loader import _SH_C0
        large_neg = np.array([[-10.0, -10.0, -10.0]], dtype=np.float32)
        rgb = np.clip(large_neg * _SH_C0 + 0.5, 0, 1)
        np.testing.assert_allclose(rgb, [[0.0, 0.0, 0.0]], atol=1e-6)


class TestCovarianceComputation:
    """Tests for 3-D covariance matrix computation."""

    def test_identity_rotation_gives_diagonal(self):
        from visualizer.loader import _compute_covariances
        scales = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
        # Identity quaternion [w=1, x=0, y=0, z=0]
        rotations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        cov = _compute_covariances(scales, rotations)[0]
        expected = np.diag([4.0, 9.0, 16.0])  # scales^2
        np.testing.assert_allclose(cov, expected, atol=1e-5)

    def test_covariance_is_symmetric(self):
        from visualizer.loader import _compute_covariances
        rng = np.random.default_rng(42)
        scales = rng.uniform(0.1, 2.0, (10, 3)).astype(np.float32)
        q_raw = rng.standard_normal((10, 4)).astype(np.float32)
        rotations = q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True)
        cov = _compute_covariances(scales, rotations)
        diff = np.abs(cov - cov.transpose(0, 2, 1))
        assert diff.max() < 1e-5

    def test_covariance_trace_equals_sum_of_squared_scales(self):
        from visualizer.loader import _compute_covariances
        scales = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rotations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        cov = _compute_covariances(scales, rotations)[0]
        expected_trace = (scales ** 2).sum()
        np.testing.assert_allclose(np.trace(cov), expected_trace, atol=1e-5)


# ---------------------------------------------------------------------------
# projector tests
# ---------------------------------------------------------------------------


class TestProjector:
    """Tests for visualizer.projector.project_gaussians."""

    def _make_camera(self, pos, look_at):
        """Return (camera_pos, wxyz) for a camera at `pos` looking at `look_at`."""
        from visualizer.projector import camera_look_at_to_wxyz
        wxyz = camera_look_at_to_wxyz(np.array(pos), np.array(look_at))
        return np.array(pos, dtype=np.float64), wxyz

    def test_point_in_front_has_positive_depth(self):
        from visualizer.projector import project_gaussians
        # Camera at origin looking toward +Z (scene center at (0,0,5)).
        cam_pos, wxyz = self._make_camera([0, 0, 0], [0, 0, 1])
        pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
        _, depths, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert valid[0], "Point in front of camera must be valid"
        assert depths[0] > 0

    def test_point_behind_camera_invalid(self):
        from visualizer.projector import project_gaussians
        cam_pos, wxyz = self._make_camera([0, 0, 0], [0, 0, 1])
        pts = np.array([[0.0, 0.0, -5.0]], dtype=np.float32)
        _, _, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert not valid[0]

    def test_centre_point_projects_to_screen_centre(self):
        from visualizer.projector import project_gaussians
        cam_pos, wxyz = self._make_camera([0, 0, -5], [0, 0, 0])
        # Point exactly at look-at target.
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        screen_xy, _, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert valid[0]
        np.testing.assert_allclose(screen_xy[0], [0.5, 0.5], atol=1e-4)

    def test_right_of_centre_has_screen_x_greater_than_half(self):
        """Viser camera at (0,0,-5) looking at origin: the camera's image-right
        direction in world space is camera +X = cross(y, z).  For this camera
        (looking in +Z with image-up = +Y world) that is world -X, so the world
        point (-1, 0, 0) should appear on the right side of the screen."""
        from visualizer.projector import project_gaussians
        cam_pos, wxyz = self._make_camera([0, 0, -5], [0, 0, 0])
        pts = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        screen_xy, _, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert valid[0]
        assert screen_xy[0, 0] > 0.5, "Camera-right point should have screen_x > 0.5"

    def test_below_centre_has_screen_y_greater_than_half(self):
        """In viser's OpenCV convention, camera +Y points down in the image.
        For this camera (looking in +Z with image-up = +Y world), the camera
        +Y direction in world is (0, -1, 0) = world -Y (down).  A world point
        at (0, -1, 0) is in the world-down direction and should appear below
        centre (screen_y > 0.5, since y=0 is the top of the image)."""
        from visualizer.projector import project_gaussians
        cam_pos, wxyz = self._make_camera([0, 0, -5], [0, 0, 0])
        pts = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
        screen_xy, _, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert valid[0]
        assert screen_xy[0, 1] > 0.5, "World-down point should appear below centre (screen_y > 0.5)"

    def test_screen_coords_in_unit_range_for_on_screen_points(self):
        from visualizer.projector import project_gaussians
        cam_pos, wxyz = self._make_camera([0, 0, -5], [0, 0, 0])
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        screen_xy, _, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert (screen_xy[valid] >= 0).all()
        assert (screen_xy[valid] <= 1).all()

    def test_depth_equals_distance_along_camera_z(self):
        from visualizer.projector import project_gaussians
        # Camera at (0,0,-10) looking along +Z.
        cam_pos, wxyz = self._make_camera([0, 0, -10], [0, 0, 0])
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # scene origin
        _, depths, valid = project_gaussians(pts, cam_pos, wxyz, math.pi / 3, 1.0)
        assert valid[0]
        np.testing.assert_allclose(depths[0], 10.0, atol=0.05)


# ---------------------------------------------------------------------------
# selector tests
# ---------------------------------------------------------------------------


class TestSurfaceSelector:
    """Tests for visualizer.selector.select_surface_gaussians."""

    def _camera_looking_at_origin(self):
        from visualizer.projector import camera_look_at_to_wxyz
        pos = np.array([0.0, 0.0, -10.0])
        wxyz = camera_look_at_to_wxyz(pos, np.zeros(3))
        return pos, wxyz

    def test_selects_nothing_for_empty_rect(self):
        from visualizer.selector import select_surface_gaussians
        cam_pos, wxyz = self._camera_looking_at_origin()
        positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
        opacities = np.array([0.9, 0.9], dtype=np.float32)
        # Rect outside screen
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, math.pi / 3, 1.0,
            screen_rect=(0.8, 0.8, 0.9, 0.9),
        )
        assert not sel.any(), "Nothing should be selected when rect misses all Gaussians"

    def test_opaque_front_gaussian_selected(self):
        """A single very opaque Gaussian at the front should be selected."""
        from visualizer.selector import select_surface_gaussians
        cam_pos, wxyz = self._camera_looking_at_origin()
        # One opaque Gaussian right at the centre.
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        opacities = np.array([0.99], dtype=np.float32)
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, math.pi / 3, 1.0,
            screen_rect=(0.0, 0.0, 1.0, 1.0),
        )
        assert sel[0], "Opaque Gaussian at centre should be selected"

    def test_occluded_gaussian_not_selected(self):
        """A Gaussian fully hidden behind an opaque front layer should not be selected."""
        from visualizer.selector import select_surface_gaussians
        cam_pos, wxyz = self._camera_looking_at_origin()

        # Two Gaussians stacked along the camera ray at the same screen pixel.
        # Front: very opaque (opacity=1.0); Back: should be occluded.
        # Place them at same x,y but different depths.
        positions = np.array(
            [[0.0, 0.0, 0.0],   # front (closer to cam at z=-10, depth=10)
             [0.0, 0.0, 2.0]],  # back (depth=12)
            dtype=np.float32
        )
        # Using full opacity to ensure front blocks back completely.
        opacities = np.array([1.0, 1.0], dtype=np.float32)
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, math.pi / 3, 1.0,
            screen_rect=(0.0, 0.0, 1.0, 1.0),
            opacity_threshold=0.9,
        )
        assert sel[0], "Front Gaussian must be selected"
        assert not sel[1], "Back Gaussian should be occluded by fully-opaque front"

    def test_transparent_front_lets_back_through(self):
        """A nearly-transparent front Gaussian should let the back one through."""
        from visualizer.selector import select_surface_gaussians
        cam_pos, wxyz = self._camera_looking_at_origin()
        positions = np.array(
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 2.0]],
            dtype=np.float32
        )
        # Very low opacity front – transmittance barely changes.
        opacities = np.array([0.05, 0.9], dtype=np.float32)
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, math.pi / 3, 1.0,
            screen_rect=(0.0, 0.0, 1.0, 1.0),
            opacity_threshold=0.9,
        )
        assert sel[0], "Transparent front Gaussian should be selected"
        assert sel[1], "Back Gaussian should be visible through transparent front"

    def test_selection_restricted_to_rect(self):
        """Gaussians outside the selection rectangle must not be selected."""
        from visualizer.selector import select_surface_gaussians
        from visualizer.projector import project_gaussians

        cam_pos, wxyz = self._camera_looking_at_origin()
        fov_y = math.pi / 3
        aspect = 1.0

        # Place one Gaussian at screen centre and one far to the side.
        positions = np.array(
            [[0.0, 0.0, 0.0],    # projects near (0.5, 0.5)
             [5.0, 0.0, 0.0]],   # projects far right
            dtype=np.float32
        )
        opacities = np.array([0.9, 0.9], dtype=np.float32)

        # Narrow rect around centre only.
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, fov_y, aspect,
            screen_rect=(0.3, 0.3, 0.7, 0.7),
        )
        assert sel[0], "Centre Gaussian should be inside rect"
        assert not sel[1], "Side Gaussian should be outside rect"

    def test_returns_correct_shape(self):
        from visualizer.selector import select_surface_gaussians
        cam_pos, wxyz = self._camera_looking_at_origin()
        N = 50
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((N, 3)).astype(np.float32)
        positions[:, 2] = np.abs(positions[:, 2]) + 1.0
        opacities = rng.uniform(0, 1, N).astype(np.float32)
        sel = select_surface_gaussians(
            positions, opacities, cam_pos, wxyz, math.pi / 3, 1.0,
            screen_rect=(0.0, 0.0, 1.0, 1.0),
        )
        assert sel.shape == (N,)
        assert sel.dtype == bool

    def test_bear_dataset_selects_plausible_count(self):
        """On the real bear dataset, full-screen selection should pick a surface layer."""
        from visualizer.loader import load_gaussians
        from visualizer.selector import select_surface_gaussians
        from visualizer.projector import camera_look_at_to_wxyz

        data = load_gaussians(BEAR_PLY)
        cam_pos = np.array([0.0, 0.5, -4.5])
        wxyz = camera_look_at_to_wxyz(cam_pos, np.array([0.0, 1.0, 3.0]))

        sel = select_surface_gaussians(
            data["positions"],
            data["opacities"],
            cam_pos,
            wxyz,
            fov_y=math.pi / 3,
            aspect=985 / 729,
            screen_rect=(0.0, 0.0, 1.0, 1.0),
        )
        count = int(sel.sum())
        total = len(data["positions"])
        # Surface layer should be a minority of all Gaussians but still substantial.
        assert count > 100, f"Expected >100 surface Gaussians, got {count}"
        assert count < total * 0.5, (
            f"Surface layer {count} is too large relative to total {total}; "
            "check opacity accumulation logic"
        )
