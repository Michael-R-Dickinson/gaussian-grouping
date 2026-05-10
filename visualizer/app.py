"""Interactive Gaussian Splatting visualizer powered by viser.

Launch with:
    python -m visualizer.app
or:
    python -m visualizer.app --ply path/to/point_cloud.ply --port 8080
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation

from .loader import DEFAULT_PLY, load_gaussians
from .projector import camera_look_at_to_wxyz
from .selector import select_surface_gaussians

# Bear scene approximate centre derived from training camera coverage.
_BEAR_SCENE_CENTER = np.array([0.0, 1.0, 3.0], dtype=np.float64)
_BEAR_CAMERA_DEFAULT_POS = np.array([0.0, 0.5, -4.5], dtype=np.float64)

# Orange colour for selected Gaussians (float [0, 1]).
_ORANGE = np.array([1.0, 0.5, 0.0], dtype=np.float32)


class GaussianVisualizer:
    """Interactive viser-based viewer for a Gaussian Splatting scene."""

    def __init__(self, ply_path: str | Path = DEFAULT_PLY, port: int = 8080):
        self._ply_path = Path(ply_path)
        self._port = port

        print(f"Loading Gaussians from {self._ply_path} …")
        data = load_gaussians(self._ply_path)
        self._positions: np.ndarray = data["positions"]   # (N, 3)
        self._opacities: np.ndarray = data["opacities"]   # (N,)
        self._base_colors: np.ndarray = data["colors"]    # (N, 3) float [0,1]
        self._covariances: np.ndarray = data["covariances"]  # (N, 3, 3)
        N = len(self._positions)
        print(f"Loaded {N:,} Gaussians.")

        self._selected_mask: np.ndarray = np.zeros(N, dtype=bool)
        self._selection_mode: bool = False

        # Per-client camera-lock state (keyed by client_id).
        self._lock_state: dict[int, dict] = {}
        self._lock_mutex = threading.Lock()

        self._server = viser.ViserServer(port=port)
        self._splat_handle: viser.GaussianSplatHandle | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._build_scene()
        self._build_gui()
        self._register_callbacks()
        print(
            f"Visualizer running at http://localhost:{self._port}\n"
            "  • WASD / touchpad to navigate\n"
            "  • Toggle 'Selection Mode', then Shift+drag to select surface Gaussians\n"
            "  • Press 'Clear Selection' to deselect all"
        )
        self._server.sleep_forever()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        # Bear dataset uses +Y-up world coordinates (COLMAP convention).
        self._server.scene.set_up_direction("+y")
        self._splat_handle = self._server.scene.add_gaussian_splats(
            name="gaussians",
            centers=self._positions,
            covariances=self._covariances,
            rgbs=self._base_colors,
            opacities=self._opacities[:, np.newaxis],
        )

    def _update_colors(self) -> None:
        """Push current colors (base + orange overlay for selected) to viser."""
        if self._splat_handle is None:
            return
        colors = self._base_colors.copy()
        if np.any(self._selected_mask):
            colors[self._selected_mask] = _ORANGE
        self._splat_handle.rgbs = colors

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        with self._server.gui.add_folder("Selection"):
            self._btn_toggle = self._server.gui.add_button(
                "Enter Selection Mode", color="blue"
            )
            self._lbl_status = self._server.gui.add_markdown(
                "_Selection mode: **off**_"
            )
            self._lbl_count = self._server.gui.add_markdown(
                "_Selected: **0** Gaussians_"
            )
            self._btn_clear = self._server.gui.add_button(
                "Clear Selection", color="red"
            )

        with self._server.gui.add_folder("View"):
            self._btn_reset_camera = self._server.gui.add_button("Reset Camera")
            self._dd_up = self._server.gui.add_dropdown(
                "World Up",
                options=["+y", "-y", "+z", "-z", "+x", "-x"],
                initial_value="+y",
            )
            self._btn_roll_left = self._server.gui.add_button("Roll Left 90°")
            self._btn_roll_right = self._server.gui.add_button("Roll Right 90°")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        @self._btn_toggle.on_click
        def _toggle_selection_mode(_) -> None:
            self._selection_mode = not self._selection_mode
            if self._selection_mode:
                self._btn_toggle.label = "Exit Selection Mode"
                self._btn_toggle.color = "orange"
                self._lbl_status.content = "_Selection mode: **on** — viewport locked_"
                # Lock all currently connected clients.
                for client in self._server.get_clients().values():
                    self._lock_camera(client)
            else:
                self._btn_toggle.label = "Enter Selection Mode"
                self._btn_toggle.color = "blue"
                self._lbl_status.content = "_Selection mode: **off**_"
                # Unlock all clients.
                with self._lock_mutex:
                    self._lock_state.clear()

        @self._btn_clear.on_click
        def _clear_selection(_) -> None:
            self._selected_mask[:] = False
            self._lbl_count.content = "_Selected: **0** Gaussians_"
            self._update_colors()

        @self._btn_reset_camera.on_click
        def _reset_camera(_) -> None:
            for client in self._server.get_clients().values():
                _set_default_camera(client)

        @self._dd_up.on_update
        def _on_up_change(event: viser.GuiUpdateEvent) -> None:
            self._server.scene.set_up_direction(event.target.value)

        @self._btn_roll_left.on_click
        def _roll_left(_) -> None:
            _roll_all_clients(self._server, -90.0)

        @self._btn_roll_right.on_click
        def _roll_right(_) -> None:
            _roll_all_clients(self._server, +90.0)

        # Rectangle select (shift+drag).
        @self._server.scene.on_rect_select(modifier="shift")
        def _on_rect_select(event: viser.SceneRectSelectEvent) -> None:
            if not self._selection_mode:
                return
            client = event.client
            cam = client.camera
            new_sel = select_surface_gaussians(
                positions=self._positions,
                opacities=self._opacities,
                camera_pos=np.array(cam.position, dtype=np.float64),
                camera_wxyz=np.array(cam.wxyz, dtype=np.float64),
                fov_y=float(cam.fov),
                aspect=float(cam.aspect),
                screen_rect=(
                    event.screen_min[0],
                    event.screen_min[1],
                    event.screen_max[0],
                    event.screen_max[1],
                ),
            )
            self._selected_mask |= new_sel
            count = int(np.sum(self._selected_mask))
            self._lbl_count.content = f"_Selected: **{count:,}** Gaussians_"
            self._update_colors()

        # Set default camera for each new client.
        @self._server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            _set_default_camera(client)

            @client.camera.on_update
            def _on_cam_update(_) -> None:
                self._enforce_camera_lock(client)

    def _lock_camera(self, client: viser.ClientHandle) -> None:
        with self._lock_mutex:
            self._lock_state[client.client_id] = {
                "position": np.array(client.camera.position),
                "wxyz": np.array(client.camera.wxyz),
                "look_at": np.array(client.camera.look_at),
                "up_direction": np.array(client.camera.up_direction),
                "updating": False,
            }

    def _enforce_camera_lock(self, client: viser.ClientHandle) -> None:
        with self._lock_mutex:
            state = self._lock_state.get(client.client_id)
            if state is None or state["updating"]:
                return
            # Check if camera actually moved.
            if (
                np.allclose(client.camera.position, state["position"], atol=1e-5)
                and np.allclose(client.camera.wxyz, state["wxyz"], atol=1e-5)
            ):
                return
            state["updating"] = True

        try:
            client.camera.position = state["position"]
            client.camera.wxyz = state["wxyz"]
        finally:
            with self._lock_mutex:
                if client.client_id in self._lock_state:
                    self._lock_state[client.client_id]["updating"] = False


def _roll_all_clients(server: viser.ViserServer, angle_deg: float) -> None:
    """Roll every connected client's camera around its view axis (local +Z)."""
    roll = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(angle_deg)])
    for client in server.get_clients().values():
        w, x, y, z = client.camera.wxyz
        r_c2w = Rotation.from_quat([x, y, z, w])
        q = (r_c2w * roll).as_quat()  # [x, y, z, w]
        client.camera.wxyz = np.array([q[3], q[0], q[1], q[2]])


def _set_default_camera(client: viser.ClientHandle) -> None:
    wxyz = camera_look_at_to_wxyz(
        camera_pos=_BEAR_CAMERA_DEFAULT_POS,
        look_at=_BEAR_SCENE_CENTER,
    )
    client.camera.wxyz = wxyz
    client.camera.position = _BEAR_CAMERA_DEFAULT_POS
    client.camera.look_at = _BEAR_SCENE_CENTER


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian Splatting visualizer")
    parser.add_argument(
        "--ply",
        type=Path,
        default=DEFAULT_PLY,
        help="Path to a trained point_cloud.ply",
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    GaussianVisualizer(ply_path=args.ply, port=args.port).run()


if __name__ == "__main__":
    main()
