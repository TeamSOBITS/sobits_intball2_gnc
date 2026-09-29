#!/usr/bin/env python3
"""Compare a ``capture_stereo_depth.py`` capture against ground-truth depth raycast into the OctoMap.

No ROS: loads the ``.npz``, renders the depth the left camera should see in
``jem_octomap.bt`` from the saved pose, and reports stereo error per
ground-truth distance bin (bias, |error| percentiles, holes, points much
closer than the truth, points behind the camera).

Ground truth uses the virtual camera's renderer (exact voxel traversal) and
is only as good as the map: 0.05 m voxels, JEM crop only (rays leaving the
crop have no truth), and the OctoMap may differ from Gazebo's mesh.

Usage:
    python3 test/manual/compare_stereo_depth_to_octomap.py /tmp/stereo_depth.npz [--stride 4] [--save-png /tmp/err.png]
"""
import argparse
import os

import numpy as np

import sobits_intball2_gnc_cpp

DEFAULT_MAP = os.path.join(os.path.dirname(__file__), "..", "..", "maps", "jem_octomap.bt")
# body -> cameraL_link (intball2_programs/urdf/ib2.urdf) -> cameraL_optical_frame (stereo_pointcloud.launch.py)
CAMERA_L_IN_BODY = np.array([0.025, 0.14, 0.0])
BODY_R_OPTICAL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
BINS_M = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
CLOSER_THAN_TRUTH_M = 0.3


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class OctomapRenderer:
    """The virtual camera's C++ voxel traversal (DepthRenderer) over the OctoMap, as ground truth."""

    def __init__(self, map_file):
        resolution, flat = sobits_intball2_gnc_cpp.load_octomap_points(map_file)
        self.renderer = sobits_intball2_gnc_cpp.DepthRenderer()
        self.renderer.set_static(flat, resolution)


def ground_truth_depth(truth, k, shape, body_pos, body_quat, stride, threads=4):
    """Optical-axis depth (z) for pixels sampled every ``stride``, as seen from the left camera; NaN = no hit."""
    r_iss_body = quat_to_matrix(body_quat)
    cam_origin = np.asarray(body_pos) + r_iss_body @ CAMERA_L_IN_BODY
    depth = truth.renderer.render(list(map(float, cam_origin)), (r_iss_body @ BODY_R_OPTICAL).ravel().tolist(),
                                  k[0, 0], k[1, 1], k[0, 2], k[1, 2], shape[1], shape[0], threads=threads)
    rows, cols = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
    return depth[rows, cols].astype(float), rows, cols


def report(stereo, truth):
    has_truth = np.isfinite(truth)
    print(f"pixels: {truth.size}, with truth {has_truth.mean():.1%}, stereo valid {np.isfinite(stereo).mean():.1%}, "
          f"stereo z<0 {(stereo < 0).mean():.1%}")
    both = has_truth & np.isfinite(stereo) & (stereo > 0)
    err = stereo[both] - truth[both]
    if err.size:
        print(f"median truth {np.nanmedian(truth):.3f}m, stereo {np.median(stereo[both]):.3f}m, error {np.median(err):+.3f}m; "
              f"farther >0.1/0.3/1m {(err > 0.1).mean():.2%}/{(err > 0.3).mean():.2%}/{(err > 1).mean():.2%}, "
              f"closer >0.1m {(err < -0.1).mean():.2%}, max stereo z {stereo[both].max():.2f}m")
    print(f"{'truth bin [m]':>14} {'n':>7} {'hole':>6} {'z<0':>5} {'closer>' + str(CLOSER_THAN_TRUTH_M):>11} "
          f"{'bias':>7} {'|e| p50':>8} {'|e| p90':>8}")
    for lo, hi in zip(BINS_M[:-1], BINS_M[1:]):
        sel = has_truth & (truth >= lo) & (truth < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        s, t = stereo[sel], truth[sel]
        valid = np.isfinite(s) & (s > 0)
        err = s[valid] - t[valid]
        closer = (s > 0) & (s < t - CLOSER_THAN_TRUTH_M)
        stats = (f"{np.median(err):7.3f} {np.percentile(np.abs(err), 50):8.3f} {np.percentile(np.abs(err), 90):8.3f}"
                 if err.size else f"{'-':>7} {'-':>8} {'-':>8}")
        print(f"{lo:6.1f}-{hi:<6.1f} {n:7d} {1 - np.isfinite(s).mean():6.1%} {(s < 0).mean():5.1%} "
              f"{closer.mean():11.1%} {stats}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frame", type=int, default=-1, help="which captured frame (default: last)")
    parser.add_argument("--save-png", default=None)
    args = parser.parse_args()

    cap = np.load(args.capture)
    depth = cap["depth"][args.frame]
    truth, rows, cols = ground_truth_depth(
        OctomapRenderer(args.map), cap["k"], depth.shape, cap["body_pos"], cap["body_quat"], args.stride)
    stereo = depth[rows, cols]
    print(f"capture {args.capture}: frame {args.frame} of {len(cap['depth'])}, stamp {cap['stamps'][args.frame]:.2f}s, "
          f"frame_id {cap['frame_id']}, stride {args.stride}")
    report(stereo, truth)

    if args.save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title, kw in [
                (axes[0], truth, "truth z [m]", dict(vmin=0, vmax=6)),
                (axes[1], stereo, "stereo z [m]", dict(vmin=0, vmax=6)),
                (axes[2], stereo - truth, "stereo - truth [m]", dict(vmin=-1, vmax=1, cmap="coolwarm"))]:
            ax.imshow(img, **kw)
            ax.set_title(title)
            fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
        fig.savefig(args.save_png, dpi=80, bbox_inches="tight")
        print(f"saved {args.save_png}")


if __name__ == "__main__":
    main()
