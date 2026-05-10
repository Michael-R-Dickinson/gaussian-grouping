---
name: Gaussian Splatting Visualizer
description: Interactive viser-based visualizer for the bear Gaussian Splatting scene with surface selection
type: project
---

Interactive visualizer in `visualizer/` using viser 1.0.27.

**Why:** User wanted an interactive viewer for trained Gaussian Splatting scenes with a surface Gaussian selector.

**How to apply:** When modifying or extending the visualizer, refer to these architectural decisions.

## Architecture
- `visualizer/loader.py`: Load PLY, compute 3D covariances (Σ = R S² Rᵀ), DC SH → RGB
- `visualizer/projector.py`: Camera projection math; `camera_look_at_to_wxyz` replicates viser's `_update_wxyz` (uses 180° rotation around forward axis to flip up → camera +Y = down)
- `visualizer/selector.py`: Grid-based surface selection (24×24 cells, opacity accumulation threshold=0.9)
- `visualizer/app.py`: Main viser server; `GaussianVisualizer` class
- Launch: `python -m visualizer` from project root

## Key viser conventions
- `client.camera.wxyz` = camera-to-world quaternion [w,x,y,z]; camera +Z=forward, +Y=down, +X=image-right
- `on_rect_select(modifier="shift")` gives `screen_min/screen_max` in OpenCV coords (0,0=upper-left)
- `add_gaussian_splats()` takes `rgbs` float [0,1] and `opacities` (N,1) float [0,1]
- Screen_x = (ndc_x_python + 1) / 2 and screen_y = (ndc_y_python + 1) / 2 match viser events
- Scene uses `set_up_direction("+y")` for bear dataset (COLMAP +Y-up convention)

## Camera locking
Selection mode locks viewport by saving camera state on mode entry and restoring on each `on_update` callback. Uses `updating` flag to prevent recursion.

## Tests
29 unit tests in `tests/test_visualizer.py`. Run with `python -m pytest tests/test_visualizer.py`.
Cover: PLY loading, SH conversion, covariance math, projection geometry, surface selection algorithm, bear dataset integration test.
