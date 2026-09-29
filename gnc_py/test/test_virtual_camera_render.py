"""DepthRenderer (virtual camera core) against geometry with known depth."""
import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
W = H = 41
F = 20.0
C = 20.0


def render(renderer, **kwargs):
    return renderer.render([0.0, 0.0, 0.0], IDENTITY, F, F, C, C, W, H, **kwargs)


def wall_points(z, half=2.0, resolution=0.05):
    """Voxel centers of a one-voxel-thick wall facing the camera at depth ~z."""
    ticks = np.arange(-half, half, resolution) + 0.5 * resolution
    xs, ys = np.meshgrid(ticks, ticks)
    zc = (np.floor(z / resolution) + 0.5) * resolution
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, zc)], axis=1).ravel().tolist()


def test_empty_scene_is_nan():
    assert np.isnan(render(sobits_intball2_gnc_cpp.DepthRenderer())).all()


def test_static_wall_depth_is_its_near_face():
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    renderer.set_static(wall_points(1.0), 0.05)
    depth = render(renderer)
    assert depth[20, 20] == pytest.approx(1.0, abs=1e-6)
    # Optical-axis depth, not ray length: the same flat wall reads the same z off-axis.
    assert depth[5, 35] == pytest.approx(1.0, abs=1e-6)


def test_box_occludes_wall_and_max_range_cuts():
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    renderer.set_static(wall_points(2.0), 0.05)
    box = ([0.0, 0.0, 1.0], [0.2, 0.2, 0.2], IDENTITY)
    depth = render(renderer, boxes=[box])
    assert depth[20, 20] == pytest.approx(0.8, abs=1e-6)
    assert depth[20, 0] == pytest.approx(2.0, abs=1e-6)
    assert np.isnan(render(renderer, boxes=[box], max_range=0.5)).all()


def test_shape_instance_is_placed_and_occludes():
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    corners = [[sx * 0.05, sy * 0.05, sz * 0.05] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    cube = renderer.add_shape(np.ravel(corners).tolist(), 0.1)
    rotated = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    depth = render(renderer, instances=[(cube, rotated, [0.0, 0.0, 1.5])],
                   boxes=[([0.0, 0.0, 3.0], [1.0, 1.0, 0.1], IDENTITY)])
    assert depth[20, 20] == pytest.approx(1.4, abs=1e-6)
    assert depth[20, 15] == pytest.approx(2.9, abs=1e-6)
