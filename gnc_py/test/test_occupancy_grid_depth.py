"""OccupancyGrid depth layer: log-odds updates, clearing, inflation bookkeeping, threading."""
import itertools
import math
import threading

import numpy as np
import pytest

sobits_intball2_gnc_cpp = pytest.importorskip("sobits_intball2_gnc_cpp")

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# Optical z -> grid +x, optical x -> grid -y, optical y -> grid -z.
LOOK_X = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
RES = 0.1
P_HIT, P_MISS, P_MIN, P_MAX, P_OCC = 0.65, 0.35, 0.12, 0.90, 0.80


def logit(p):
    return np.float32(math.log(p / (1.0 - p)))


def step(value, hit):
    """float32 replica of one per-frame update, occupied iff value >= logit(p_occ)."""
    value = np.float32(value + (logit(P_HIT) if hit else logit(P_MISS)))
    return np.float32(min(max(value, logit(P_MIN)), logit(P_MAX)))


def frames_until(value, hit, want_occupied):
    n = 0
    while (value >= logit(P_OCC)) != want_occupied:
        value = step(value, hit)
        n += 1
    return n


def single_ray_grid(inflation=0.0):
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, inflation)
    grid.enable_depth_layer([0.0, 0.0, 0.0], [3.5, 1.0, 1.0])
    return grid


def one_pixel(grid, value, origin=(0.05, 0.55, 0.55)):
    return grid.integrate_depth(np.array([[value]], np.float32), 1.0, 1.0, 0.0, 0.0, LOOK_X, list(origin))


def test_single_ray_occupies_end_and_clears_when_seen_through():
    grid = single_ray_grid()
    end = [2.05, 0.55, 0.55]
    n_occ = frames_until(logit(P_MIN), True, True)
    assert n_occ == 6
    for k in range(n_occ):
        assert not grid.depth_occupied(end)
        one_pixel(grid, 2.0)
    assert grid.depth_occupied(end)
    assert grid.inflated_occupied(end)
    for x in np.arange(0.05, 2.0, RES):
        assert not grid.depth_occupied([x, 0.55, 0.55])
    assert grid.depth_occupied_cells().shape == (1, 3)
    np.testing.assert_allclose(grid.depth_occupied_cells()[0], end, atol=1e-9)

    value = logit(P_MIN)
    for _ in range(n_occ):
        value = step(value, True)
    n_free = frames_until(value, False, False)
    for k in range(n_free):
        assert grid.depth_occupied(end)
        one_pixel(grid, 3.0)
    assert not grid.depth_occupied(end)
    assert grid.depth_occupied([3.05, 0.55, 0.55]) is False


def test_inf_clears_without_hit_and_invalid_pixels_are_ignored():
    grid = single_ray_grid()
    for _ in range(10):
        one_pixel(grid, 2.0)
    assert grid.depth_occupied([2.05, 0.55, 0.55])
    for bad in (-np.inf, np.nan, 0.2):
        assert one_pixel(grid, bad) == (0, 0, 0)
    assert grid.depth_occupied([2.05, 0.55, 0.55])

    for _ in range(10):
        points, updated, _ = one_pixel(grid, np.inf)
        assert points == 1 and updated == 31  # cells 0..30: max_range 3.0 m from x=0.05
    assert not grid.depth_occupied([2.05, 0.55, 0.55])
    assert grid.depth_occupied_cells().shape == (0, 3)
    # A beyond-range finite depth also clears only.
    one_pixel(grid, 5.0)
    assert grid.depth_occupied_cells().shape == (0, 3)


def test_static_layer_survives_depth_misses():
    grid = single_ray_grid(inflation=0.1)
    grid.add_box([1.05, 0.55, 0.55], [0.01, 0.01, 0.01])
    before = [grid.inflated_occupied([x, 0.55, 0.55]) for x in np.arange(0.05, 3.5, RES)]
    for _ in range(20):
        one_pixel(grid, np.inf)
    after = [grid.inflated_occupied([x, 0.55, 0.55]) for x in np.arange(0.05, 3.5, RES)]
    assert before == after
    assert grid.inflated_occupied([1.05, 0.55, 0.55])
    assert not grid.depth_occupied([1.05, 0.55, 0.55])


def cell_centers(lower, upper):
    lo = np.floor(np.asarray(lower) / RES).astype(int)
    hi = np.floor(np.asarray(upper) / RES).astype(int)
    return lo, hi, [(np.array(c) + 0.5) * RES
                    for c in itertools.product(*[range(a, b + 1) for a, b in zip(lo, hi)])]


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_inflation_counter_matches_brute_force():
    rng = np.random.default_rng(3)
    lower, upper = [-1.0, -1.0, -0.5], [1.0, 1.0, 0.5]
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.2)
    grid.enable_depth_layer(lower, upper)
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    lo, hi, centers = cell_centers(lower, upper)
    k = math.ceil((0.2 - 1e-5) / RES)
    w = h = 48
    f = 24.0
    boxes = []
    for pose in range(12):
        if pose % 4 == 0:
            boxes = [(list(rng.uniform(-1.5, 1.5, 3)), list(rng.uniform(0.05, 0.4, 3)),
                      list(random_rotation(rng).ravel())) for _ in range(3)]
        rot = random_rotation(rng)
        origin = rng.uniform(lower, upper)
        depth = renderer.render(list(origin), list(rot.ravel()), f, f, w / 2, h / 2, w, h,
                                boxes=boxes, max_range=3.0)
        depth[np.isnan(depth)] = np.inf
        for _ in range(int(rng.integers(1, 9))):
            grid.integrate_depth(depth, f, f, w / 2, h / 2, rot, list(origin))

    occupied = {tuple(np.floor(c / RES).astype(int)) for c in grid.depth_occupied_cells()}
    assert occupied
    for c in centers:
        cell = np.floor(c / RES).astype(int)
        expected = any(tuple(cell + d) in occupied for d in itertools.product(range(-k, k + 1), repeat=3))
        assert grid.inflated_occupied(list(c)) == expected, c
    # Outside the box the depth layer contributes nothing.
    assert not grid.inflated_occupied([upper[0] + 0.25, 0.0, 0.0])


def test_occupied_cells_lie_on_rendered_box_surface():
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    center, half = np.array([2.0, 0.1, -0.05]), np.array([0.3, 0.4, 0.25])
    box = (list(center), list(half), IDENTITY)
    origin = [0.0, 0.0, 0.0]
    w = h = 100
    f, c = 60.0, 49.5
    depth = renderer.render(origin, LOOK_X, f, f, c, c, w, h, boxes=[box], max_range=3.0)
    depth[np.isnan(depth)] = np.inf
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.0)
    grid.enable_depth_layer([-0.5, -1.5, -1.5], [3.0, 1.5, 1.5])
    for _ in range(8):
        grid.integrate_depth(depth, f, f, c, c, LOOK_X, origin)
    cells = grid.depth_occupied_cells()
    assert len(cells) > 20
    near_face_x = center[0] - half[0]
    assert np.all(np.abs(cells[:, 0] - near_face_x) <= RES)
    assert np.all(np.abs(cells[:, 1:] - center[1:]) <= half[1:] + RES)


def test_grazing_face_is_occupied():
    """A face seen nearly edge-on: rays to its far part cross the cells of its near part, so
    those cells get more misses than hits each frame (docs/archive/jaxa_baseline_gazebo_port_plan.md,
    2026-10-05). A hit must still win, or the face never enters the grid."""
    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    face_y = -0.25
    center, half = np.array([1.75, face_y - 0.3, 0.0]), np.array([0.75, 0.3, 0.25])
    box = (list(center), list(half), IDENTITY)
    origin = [0.0, 0.0, 0.0]
    w = h = 100
    f, c = 60.0, 49.5
    depth = renderer.render(origin, LOOK_X, f, f, c, c, w, h, boxes=[box], max_range=3.0)
    depth[np.isnan(depth)] = np.inf
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.0)
    grid.enable_depth_layer([-0.5, -1.5, -1.5], [3.0, 1.5, 1.5])
    for _ in range(8):
        grid.integrate_depth(depth, f, f, c, c, LOOK_X, origin)
    cells = grid.depth_occupied_cells()
    side = cells[(np.abs(cells[:, 1] - face_y) <= RES) & (cells[:, 0] >= 1.0) & (cells[:, 0] <= 2.5)]
    # The face spans 15 cells in x; far ones get no ray at this resolution, so require most of
    # the span up to its far part (the majority vote kept 4 cells near its front edge).
    columns = np.unique(np.round(side[:, 0], 3))
    assert len(columns) >= 8 and columns.max() >= 2.0


def test_without_depth_layer_matches_static_only():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(0.1, 0.2)
    grid.add_box([0.0, 0.0, 0.0], [0.15, 0.05, 0.05])
    grid.add_points([1.0, 1.0, 1.0])
    assert grid.inflated_occupied([0.0, 0.0, 0.0])
    assert grid.inflated_occupied([0.35, 0.25, 0.25])
    assert not grid.inflated_occupied([0.45, 0.0, 0.0])
    assert grid.inflated_occupied([1.25, 1.25, 1.25])
    assert not grid.inflated_occupied([1.35, 1.0, 1.0])
    assert not grid.depth_occupied([0.0, 0.0, 0.0])
    assert grid.depth_occupied_cells().shape == (0, 3)
    with pytest.raises(Exception):
        grid.integrate_depth(np.ones((1, 1), np.float32), 1.0, 1.0, 0.0, 0.0, IDENTITY, [0.0, 0.0, 0.0])


def test_concurrent_integrate_and_queries():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(RES, 0.2)
    grid.add_box([2.0, 0.0, 0.0], [0.2, 0.2, 0.2])
    grid.enable_depth_layer([-1.0, -1.5, -1.5], [4.0, 1.5, 1.5])
    n = 3
    wf = []
    for i in range(n + 1):
        wf += [4.0 * i / n, 0.0, 0.0, 0.0, 0.0, 0.0]
    ok, _ec, T, C, _d = sobits_intball2_gnc_cpp.plan_minco(wf, [0.0] * 3, [0.0] * 3, 0.0, 1.0)
    assert ok
    reference = sobits_intball2_gnc_cpp.rebound_pairs(grid, T, C, 0.3)

    renderer = sobits_intball2_gnc_cpp.DepthRenderer()
    wall = ([2.5, 1.2, 0.0], [0.1, 0.2, 0.2], IDENTITY)
    depth = renderer.render([0.0, 0.0, 0.0], LOOK_X, 30.0, 30.0, 32.0, 32.0, 64, 64, boxes=[wall], max_range=3.0)
    depth[np.isnan(depth)] = np.inf
    errors = []

    def writer():
        try:
            for _ in range(30):
                grid.integrate_depth(depth, 30.0, 30.0, 32.0, 32.0, LOOK_X, [0.0, 0.0, 0.0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(30):
                assert grid.inflated_occupied([2.0, 0.0, 0.0])
                status, pairs = sobits_intball2_gnc_cpp.rebound_pairs(grid, T, C, 0.3)
                # Pair directions may change as depth cells appear; the static collision may not vanish.
                assert status == reference[0] and len(pairs) > 0
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(grid.depth_occupied_cells()) > 0
    final = sobits_intball2_gnc_cpp.rebound_pairs(grid, T, C, 0.3)
    assert sobits_intball2_gnc_cpp.rebound_pairs(grid, T, C, 0.3) == final


def test_snapshot_is_isolated_from_later_writes():
    grid = sobits_intball2_gnc_cpp.OccupancyGrid(0.1, 0.1)
    grid.add_box([0.0, 0.0, 0.0], [0.05, 0.05, 0.05])
    grid.enable_depth_layer([-1.0, -1.0, -1.0], [3.0, 1.0, 1.0])
    snap = grid.snapshot()
    far = [2.05, 0.05, 0.05]
    for _ in range(10):
        grid.integrate_depth(np.array([[2.0]], np.float32), 1.0, 1.0, 0.0, 0.0, LOOK_X, [0.05, 0.05, 0.05])
    grid.add_box([-0.5, 0.0, 0.0], [0.05, 0.05, 0.05])
    assert grid.depth_occupied(far) and not snap.depth_occupied(far)
    assert grid.inflated_occupied([-0.5, 0.0, 0.0]) and not snap.inflated_occupied([-0.5, 0.0, 0.0])
    assert snap.inflated_occupied([0.0, 0.0, 0.0])
    snap.add_box([0.5, 0.5, 0.5], [0.05, 0.05, 0.05])
    assert not grid.inflated_occupied([0.5, 0.5, 0.5])
