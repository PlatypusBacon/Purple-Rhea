"""
Depth-Map Fusion — dense reconstruction via optical flow between views.

For each pair of frames (separated by PAIR_STEPS):
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
REPROJ_THRESH_PX = 4.0
PAIR_STEPS       = [3, 6]
DEBUG_DIR        = "output/depth_debug"


def reconstruct_depth_fusion(images, projections, masks=None):
    """
    Returns (N,3) point cloud and (N,3) BGR color array.
    """
    n = len(images)

    if masks is None:
        print(f"\n[depth] Building projection-based masks for {n} images...")
        masks = [_project_disk_mask(img, P, i) for i, (img, P) in
                 enumerate(zip(images, projections))]

    os.makedirs(DEBUG_DIR, exist_ok=True)
    _save_camera_centres(projections)

    all_pts = []
    all_col = []

    for step in PAIR_STEPS:
        print(f"\n[depth] === Pair step {step} ({step * config.STEP_DEGREES:.0f}°) ===")
        for i in range(n):
            j = (i + step) % n
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


def _save_camera_centres(projections):
    """Write camera centres to a debug PLY so you can visually check the orbit."""
    path = os.path.join(DEBUG_DIR, "cameras.ply")
    centres = []
    for P in projections:
        U, S, Vt = np.linalg.svd(P)
        C = Vt[-1, :3] / Vt[-1, 3]
        centres.append(C)
    centres = np.array(centres)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(centres)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for c in centres:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} 255 0 0\n")
    print(f"[depth] Camera centres saved to {path}")
    print(f"[depth] Centre range: X=[{centres[:,0].min():.4f},{centres[:,0].max():.4f}] "
          f"Y=[{centres[:,1].min():.4f},{centres[:,1].max():.4f}] "
          f"Z=[{centres[:,2].min():.4f},{centres[:,2].max():.4f}]")


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

    median_mag = np.median(np.sqrt(dx**2 + dy**2))
    print(f"  pair {idx1:02d}-{idx2:02d}: median flow magnitude = {median_mag:.1f} px")

    if config.DEBUG_VIZ:
        _save_flow_debug(flow, mask1, img1, idx1, idx2)

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

    keep = _reproj_filter(X3d, pts1.T, pts2.T, P1, P2, REPROJ_THRESH_PX)
    X3d = X3d[keep]
    xs_g, ys_g = xs_g[keep], ys_g[keep]

    colors = img1[ys_g.astype(int), xs_g.astype(int)]

    if len(X3d) > 0:
        dist_to_origin = np.linalg.norm(X3d, axis=1)
        print(f"  pair {idx1:02d}-{idx2:02d}: {len(xs[good_w])} corr → {len(X3d)} pts  "
              f"dist_to_origin=[{dist_to_origin.min():.4f}, {dist_to_origin.median() if False else np.median(dist_to_origin):.4f}, {dist_to_origin.max():.4f}]")
    else:
        print(f"  pair {idx1:02d}-{idx2:02d}: 0 points after reproj filter")

    return X3d, colors


def _save_flow_debug(flow, mask, img, idx1, idx2):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_bgr[mask == 0] = 0
    cv2.imwrite(os.path.join(DEBUG_DIR, f"flow_{idx1:02d}_{idx2:02d}.png"), flow_bgr)


def _project_disk_mask(img, P, frame_idx):
    """
    Project the bounding cylinder of the object volume into the image:
      - bottom ring: z = VOXEL_Z_MIN, radius = RIG_BASE_LENGTH
      - top ring:    z = VOXEL_Z_MAX, radius = RIG_BASE_LENGTH
    Fill the convex hull of all projected points as the mask.
    This captures everything that could sit on the turntable.
    """
    h_img, w_img = img.shape[:2]
    R = config.RIG_BASE_LENGTH
    z_lo = config.VOXEL_Z_MIN
    z_hi = config.VOXEL_Z_MAX
    N_SAMPLES = 360

    angles = np.linspace(0, 2 * np.pi, N_SAMPLES, endpoint=False)
    cx_ring = R * np.cos(angles)
    cy_ring = R * np.sin(angles)

    bottom = np.column_stack([cx_ring, cy_ring,
                              np.full(N_SAMPLES, z_lo),
                              np.ones(N_SAMPLES)])
    top = np.column_stack([cx_ring, cy_ring,
                           np.full(N_SAMPLES, z_hi),
                           np.ones(N_SAMPLES)])
    world_pts = np.vstack([bottom, top])

    proj = (P @ world_pts.T).T
    d = proj[:, 2]
    valid = d > 1e-6
    if valid.sum() < 5:
        print(f"  [mask {frame_idx:02d}] cylinder projection failed")
        return np.zeros((h_img, w_img), dtype=np.uint8)

    px = proj[valid, 0] / d[valid]
    py = proj[valid, 1] / d[valid]

    pts_2d = np.stack([px, py], axis=1).astype(np.float32)
    hull = cv2.convexHull(pts_2d.reshape(-1, 1, 2))

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    mask_px = int(mask.sum() // 255)
    print(f"  [mask {frame_idx:02d}] projected cylinder: "
          f"hull {len(hull)} verts, "
          f"covers {mask_px} px ({100*mask_px/(h_img*w_img):.1f}%)")

    os.makedirs("output/silhouettes", exist_ok=True)
    cv2.imwrite(f"output/silhouettes/mask_{frame_idx:02d}.png", mask)
    debug = img.copy()
    debug[mask == 0] = (debug[mask == 0] * 0.3).astype(np.uint8)
    cv2.drawContours(debug, [hull.astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.imwrite(f"output/silhouettes/debug_{frame_idx:02d}.jpg", debug)

    return mask


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
