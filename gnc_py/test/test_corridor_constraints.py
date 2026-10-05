import numpy as np

from sobits_intball2_gnc.guidance.legacy.firi.corridor_constraints import (
    local_corridor_prefix,
)


def test_local_prefix_splits_route_and_duplicates_source_planes():
    route = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    # One plane for source segment 0 and two for source segment 1.
    planes = [0, 1., 0., 0., -1., 1, 0., 1., 0., -1., 1, 0., -1., 0., -1.]
    points, remapped = local_corridor_prefix(
        route, planes, current=[.2, 0., 0.], start_segment=0,
        horizon_m=1.3, max_piece_length_m=.5,
    )
    assert np.allclose(points[-1], [1.5, 0., 0.])
    # 0.8 m of source 0 becomes two local pieces, each with its plane; the
    # next 0.5 m is source 1 and carries both of its planes.
    assert [int(remapped[i]) for i in range(0, len(remapped), 5)] == [0, 1, 2, 2]


def test_local_prefix_rejects_route_without_plane_mapping():
    route = np.array([[0., 0., 0.], [1., 0., 0.]])
    try:
        local_corridor_prefix(route, [], [0., 0., 0.], 0, 1., .75)
    except ValueError as exc:
        assert "no corresponding" in str(exc)
    else:
        raise AssertionError("missing corridor planes must be rejected")
