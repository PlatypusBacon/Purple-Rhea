"""
Depth-Map Fusion — dense reconstruction via optical flow between adjacent views.

For each consecutive pair of frames:
  1. Compute dense optical flow (Farneback)
  2. Keep correspondences that fall inside the object mask in both views
  3. Triangulate to 3D using the two projection matrices
  4. Filter by reprojection error and bounding box

All partial clouds are merged and outlier-filtered into a single (N, 3) array.
"""

import cv2
import numpy as np
import os
import config


FLOW_SUBSAMPLE   = 4
REPROJ_THRESH_PX = 3.0
PAIR_STEP        = 1


def reconstruct_depth_fusion(images, projections, masks=None):
    """
    Returns (N,3) point cloud and (N,3) BGR color array.
    """
    n = len(images)

    if masks is None:
        from pipeline.visual_hull import _build_mask
        print(f"\n[depth] Building masks for {n} images...")
        masks = [_build_mask(img, i) for i, img in enumerate(images)]
        os.makedirs("output/silhouettes", exist_ok=True)
        for i, (img, mask) in enumerate(zip(images, masks)):
            cv2.imwrite(f"output/silhouettes/mask_{i:02d}.png", mask)

    all_pts = []
    all_col = []

    for i in range(n):
        j = (i + PAIR_STEP) % n
        pts, col = _process_pair(
            images[i], images[j],
            projections[i], projections[j],
            masks[i], masks[j],
            i, j,
        )
        if len(pts) > 0:
            all_pts.append(pts)
            all_col.append(col)

    if not all_pts:
        print("[depth] WARNING: no points from any pair")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    points = np.vstack(all_pts)
    colors = np.vstack(all_col)
    print(f"\n[depth] Raw merged cloud: {len(points)} points")

    points, colors = _bbox_filter(points, colors)
    points, colors = _statistical_filter(points, colors)

    print(f"[depth] Final cloud: {len(points)} points\n")
    return points, colors


def _process_pair(img1, img2, P1, P2, mask1, mask2, idx1, idx2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )

    h_img, w_img = mask1.shape
    ys, xs = np.where(mask1 > 0)

    if len(xs) == 0:
        print(f"  pair {idx1:02d}-{idx2:02d}: empty mask, skipping")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    step = max(1, FLOW_SUBSAMPLE)
    xs, ys = xs[::step], ys[::step]

    dx = flow[ys, xs, 0]
    dy = flow[ys, xs, 1]
    xs2 = (xs + dx).astype(np.float64)
    ys2 = (ys + dy).astype(np.float64)

    valid = (xs2 >= 0) & (xs2 < w_img - 1) & (ys2 >= 0) & (ys2 < h_img - 1)
    xs, ys, xs2, ys2 = xs[valid], ys[valid], xs2[valid], ys2[valid]

    in_m2 = mask2[ys2.astype(int), xs2.astype(int)] > 0
    xs, ys, xs2, ys2 = xs[in_m2], ys[in_m2], xs2[in_m2], ys2[in_m2]

    if len(xs) < 10:
        print(f"  pair {idx1:02d}-{idx2:02d}: only {len(xs)} correspondences, skipping")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    pts1 = np.vstack([xs.astype(np.float64), ys.astype(np.float64)])
    pts2 = np.vstack([xs2, ys2])

    X4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    w_h = X4d[3]
    good_w = np.abs(w_h) > 1e-6
    X4d = X4d[:, good_w]
    pts1, pts2 = pts1[:, good_w], pts2[:, good_w]
    xs_g, ys_g = xs[good_w], ys[good_w]

    X3d = (X4d[:3] / X4d[3]).T

    # Reprojection filter
    keep = _reproj_filter(X3d, pts1.T, pts2.T, P1, P2, REPROJ_THRESH_PX)
    X3d = X3d[keep]
    xs_g, ys_g = xs_g[keep], ys_g[keep]

    colors = img1[ys_g.astype(int), xs_g.astype(int)]

    print(f"  pair {idx1:02d}-{idx2:02d}: {len(xs)} corr → {len(X3d)} points after reproj filter")
    return X3d, colors


def _reproj_filter(X3d, uv1, uv2, P1, P2, thresh):
    """Keep points whose reprojection error is below thresh in both views."""
    N = len(X3d)
    X_h = np.hstack([X3d, np.ones((N, 1))])

    errs = np.zeros(N)
    for P, uv in [(P1, uv1), (P2, uv2)]:
        proj = (P @ X_h.T).T
        d = proj[:, 2]
        safe = np.where(np.abs(d) > 1e-8, d, 1.0)
        px = proj[:, 0] / safe
        py = proj[:, 1] / safe
        errs += np.sqrt((px - uv[:, 0])**2 + (py - uv[:, 1])**2)

    mean_err = errs / 2.0
    return mean_err < thresh


def _bbox_filter(points, colors):
    r  = config.VOXEL_XY_EXTENT * 1.5
    zlo = config.VOXEL_Z_MIN
    zhi = config.VOXEL_Z_MAX
    keep = (
        (np.abs(points[:, 0]) < r) &
        (np.abs(points[:, 1]) < r) &
        (points[:, 2] > zlo) &
        (points[:, 2] < zhi)
    )
    n_before = len(points)
    points, colors = points[keep], colors[keep]
    print(f"[depth] Bbox filter: {n_before} → {len(points)} "
          f"(XY<±{r:.3f}, Z=[{zlo:.3f},{zhi:.3f}])")
    return points, colors


def _statistical_filter(points, colors, nb=20, std_ratio=2.0):
    if len(points) < nb + 1:
        return points, colors
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("[depth] sklearn not available, skipping statistical filter")
        return points, colors

    nbrs = NearestNeighbors(n_neighbors=nb + 1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_dists = dists[:, 1:].mean(axis=1)
    mu, sigma = mean_dists.mean(), mean_dists.std()
    keep = mean_dists < mu + std_ratio * sigma
    n_before = len(points)
    points, colors = points[keep], colors[keep]
    print(f"[depth] Statistical filter: {n_before} → {len(points)} "
          f"(threshold={mu + std_ratio * sigma:.6f})")
    return points, colors
