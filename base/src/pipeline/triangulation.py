"""
Triangulation — recovers 3-D point positions from matched 2-D correspondences
using the Direct Linear Transform (DLT).

For each matched pair (i, j) and each good DMatch:
  - Look up the 2-D pixel coordinates of the matched keypoints
  - Stack the projection equations from P_i and P_j
  - Solve via SVD to get the homogeneous 3-D point X
  - Divide by w to get Euclidean (x, y, z)

Filters applied per pair:
  1. Reprojection error < 4px on both source frames
  2. Cheirality: point must be in front of both cameras
  3. Triangulation angle: 2°–60° (rejects near-parallel and degenerate rays)

After all pairs are merged:
  4. Mask reprojection filter: point must land inside object mask in ≥1 frame
  5. Global cheirality: reject any point behind any camera

Input:
    matches            : dict {(i,j): [DMatch]}         from feature_matching
    projections        : list[np.ndarray shape (3,4)]    from pose_computation
    keypoints_per_frame: list[list[cv2.KeyPoint]]        from feature_detection

Output:
    points_3d: np.ndarray shape (N, 3)  — raw point cloud before error_filtering
"""
import numpy as np
import cv2
import os


def triangulate(
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    projections: list[np.ndarray],
    keypoints_per_frame: list[list[cv2.KeyPoint]],
) -> np.ndarray:
    all_points: list[np.ndarray] = []

    # Pre-compute camera centres once (used for angular check)
    cam_centres = [_cam_centre(P) for P in projections]

    for (i, j), dmatch_list in matches.items():
        P_i = projections[i]
        P_j = projections[j]
        kps_i = keypoints_per_frame[i]
        kps_j = keypoints_per_frame[j]

        pts_i = np.array([kps_i[m.queryIdx].pt for m in dmatch_list],
                         dtype=np.float64)   # (M, 2)
        pts_j = np.array([kps_j[m.trainIdx].pt for m in dmatch_list],
                         dtype=np.float64)   # (M, 2)

        if len(pts_i) == 0:
            continue

        pts4d = _triangulate_dlt(P_i, P_j, pts_i, pts_j)  # (M, 3)
        n_before = len(pts4d)

        # --- Filter 1: reprojection error ---
        err_i = reproject_error(P_i, pts4d, pts_i)
        err_j = reproject_error(P_j, pts4d, pts_j)
        good_reproj = (err_i < 4.0) & (err_j < 4.0)

        # --- Filter 2: cheirality (point in front of both cameras) ---
        X_h = np.hstack([pts4d, np.ones((len(pts4d), 1))])
        depth_i = (P_i @ X_h.T)[2]
        depth_j = (P_j @ X_h.T)[2]
        good_cheirality = (depth_i > 0) & (depth_j > 0)

        # --- Filter 3: triangulation angle ---
        C_i = cam_centres[i]
        C_j = cam_centres[j]
        rays_i = pts4d - C_i
        rays_j = pts4d - C_j
        norm_i = np.linalg.norm(rays_i, axis=1, keepdims=True)
        norm_j = np.linalg.norm(rays_j, axis=1, keepdims=True)
        # Avoid divide-by-zero for degenerate points
        safe_i = np.where(norm_i > 1e-9, norm_i, 1.0)
        safe_j = np.where(norm_j > 1e-9, norm_j, 1.0)
        rays_i_n = rays_i / safe_i
        rays_j_n = rays_j / safe_j
        cos_angle = np.einsum('ij,ij->i', rays_i_n, rays_j_n)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        good_angle = (angle_deg > 2.0) & (angle_deg < 60.0)

        good = good_reproj & good_cheirality & good_angle
        pts4d = pts4d[good]

        print(f"    pair ({i:02d},{j:02d}): triangulated {n_before} points, "
              f"kept {len(pts4d)} after filters "
              f"(reproj={good_reproj.sum()} cheirality={good_cheirality.sum()} "
              f"angle={good_angle.sum()})")

        if len(pts4d):
            all_points.append(pts4d)

    if not all_points:
        print("    WARNING: no points triangulated — check matches and projections")
        return np.zeros((0, 3), dtype=np.float64)

    raw = np.vstack(all_points)
    print(f"    Total before mask filter: {len(raw)} points")

    # --- Filter 4: mask reprojection ---
    masks = _load_masks(len(projections))
    if masks:
        raw = _filter_by_masks(raw, projections, masks)
        print(f"    After mask reprojection filter: {len(raw)} points remain")
    else:
        print("    WARNING: no masks found in output/ — skipping mask filter")

    # --- Filter 5: global cheirality across ALL cameras ---
    if len(raw):
        X_h = np.hstack([raw, np.ones((len(raw), 1))])
        behind = np.zeros(len(raw), dtype=bool)
        for P in projections:
            depths = (P @ X_h.T)[2]
            behind |= (depths <= 0)
        n_before = len(raw)
        raw = raw[~behind]
        print(f"    Global cheirality filter: {n_before} -> {len(raw)} points "
              f"(removed {behind.sum()} behind-camera points)")

    return raw


# --------------------------------------------------------------------------- #
#  Camera centre                                                               #
# --------------------------------------------------------------------------- #

def _cam_centre(P: np.ndarray) -> np.ndarray:
    """Extract camera centre C from projection matrix P (null space of P)."""
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1, :3] / Vt[-1, 3]
    return C


# --------------------------------------------------------------------------- #
#  Mask loading                                                                #
# --------------------------------------------------------------------------- #

def _load_masks(n_frames: int) -> list[np.ndarray]:
    masks = []
    for i in range(n_frames):
        path = f"output/mask_frame{i:02d}.png"
        if not os.path.exists(path):
            print(f"    WARNING: mask not found: {path}")
            return []
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"    WARNING: could not read mask: {path}")
            return []
        masks.append(m)
    print(f"    Loaded {len(masks)} binary masks from output/")
    return masks


# --------------------------------------------------------------------------- #
#  Mask reprojection filter                                                    #
# --------------------------------------------------------------------------- #

def _filter_by_masks(
    points_3d: np.ndarray,
    projections: list[np.ndarray],
    masks: list[np.ndarray],
) -> np.ndarray:
    """
    Keep a 3D point if it reprojects inside the object mask in at least one
    frame. Vectorised per-frame.
    """
    keep = np.zeros(len(points_3d), dtype=bool)
    X_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    for frame_idx, (P, mask) in enumerate(zip(projections, masks)):
        h, w = mask.shape[:2]
        proj = (P @ X_h.T).T
        depth = proj[:, 2]
        valid_depth = depth > 0
        safe_depth = np.where(valid_depth, depth, 1.0)
        px = proj[:, 0] / safe_depth
        py = proj[:, 1] / safe_depth
        ix = px.astype(int)
        iy = py.astype(int)
        in_bounds = valid_depth & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        in_mask = np.zeros(len(points_3d), dtype=bool)
        idx = np.where(in_bounds)[0]
        if len(idx):
            in_mask[idx] = mask[iy[idx], ix[idx]] > 0
        keep |= in_mask

    n_removed = len(points_3d) - keep.sum()
    print(f"    Mask filter: removed {n_removed} out-of-mask points "
          f"({100 * n_removed / max(len(points_3d), 1):.1f}%)")
    return points_3d[keep]


# --------------------------------------------------------------------------- #
#  DLT core                                                                    #
# --------------------------------------------------------------------------- #

def _triangulate_dlt(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """
    DLT triangulation for a batch of point correspondences.
    Uses OpenCV's triangulatePoints as the numerically stable DLT solver.
    pts1, pts2: (M, 2) pixel coordinates
    Returns:    (M, 3) Euclidean 3-D points
    """
    p1 = pts1.T.astype(np.float32)
    p2 = pts2.T.astype(np.float32)
    X_hom = cv2.triangulatePoints(
        P1.astype(np.float32),
        P2.astype(np.float32),
        p1, p2,
    )  # (4, M)
    X_hom /= X_hom[3:4, :]
    return X_hom[:3, :].T   # (M, 3)


def reproject_error(P: np.ndarray, X3d: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in pixels."""
    X4 = np.hstack([X3d, np.ones((len(X3d), 1))]).T
    proj = P @ X4
    proj = proj[:2] / proj[2]
    return np.linalg.norm(proj.T - pts2d, axis=1)