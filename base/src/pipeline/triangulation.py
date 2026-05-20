"""
Triangulation — recovers 3-D point positions from matched 2-D correspondences
using the Direct Linear Transform (DLT).

For each matched pair (i, j) and each good DMatch:
  - Look up the 2-D pixel coordinates of the matched keypoints
  - Stack the projection equations from P_i and P_j
  - Solve via SVD to get the homogeneous 3-D point X
  - Divide by w to get Euclidean (x, y, z)

After triangulation, each 3-D point is reprojected into every frame's saved
binary mask. Points that fall outside the mask in every single frame are
discarded — this removes points that were triangulated from border/noisy
keypoints and land outside the object region.

Input:
    matches           : dict {(i,j): [DMatch]}         from feature_matching
    projections       : list[np.ndarray shape (3,4)]    from pose_computation
    keypoints_per_frame: list[list[cv2.KeyPoint]]       from feature_detection

Output:
    points_3d: np.ndarray shape (N, 3)  — raw point cloud before filtering
"""
import numpy as np
import cv2
import os


def triangulate(
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    projections: list[np.ndarray],
    keypoints_per_frame: list[list[cv2.KeyPoint]],
) -> np.ndarray:
    """
    Triangulate all matched pairs into a single point cloud, then filter
    out points that reproject outside the object mask in every frame.
    Returns an (N, 3) float64 array.
    """
    all_points: list[np.ndarray] = []

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
        all_points.append(pts4d)
        print(f"    pair ({i:02d},{j:02d}): triangulated {len(pts4d)} points")

    if not all_points:
        print("    WARNING: no points triangulated — check matches and projections")
        return np.zeros((0, 3), dtype=np.float64)

    raw = np.vstack(all_points)
    print(f"    Total before mask filter: {len(raw)} points")

    # Load the lossless binary masks saved by feature_detection and reject
    # any 3D point that reprojects outside the mask in every frame.
    masks = _load_masks(len(projections))
    if masks:
        raw = _filter_by_masks(raw, projections, masks)
        print(f"    After mask reprojection filter: {len(raw)} points remain")
    else:
        print("    WARNING: no masks found in output/ — skipping mask filter")
        print("             (run feature_detection first, or check output/ path)")

    return raw


# --------------------------------------------------------------------------- #
#  Mask loading                                                                #
# --------------------------------------------------------------------------- #

def _load_masks(n_frames: int) -> list[np.ndarray]:
    """
    Load the lossless binary masks saved by feature_detection as PNGs.
    Returns a list of grayscale masks (0 = excluded, 255 = included).
    Returns an empty list if any mask is missing so the caller can warn.
    """
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
    frame. This is vectorised per-frame to avoid a slow Python point loop.

    Strategy: a point must land inside the mask in ANY frame to be kept.
    Using ANY (rather than ALL) is more lenient and avoids discarding valid
    points that are occluded or near the border in some views.
    """
    keep = np.zeros(len(points_3d), dtype=bool)

    # Homogeneous form — (N, 4)
    X_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    for frame_idx, (P, mask) in enumerate(zip(projections, masks)):
        h, w = mask.shape[:2]

        # Project all points with this frame's matrix — (N, 3)
        proj = (P @ X_h.T).T

        depth = proj[:, 2]
        valid_depth = depth > 0

        # Avoid divide-by-zero for points behind the camera
        safe_depth = np.where(valid_depth, depth, 1.0)
        px = proj[:, 0] / safe_depth
        py = proj[:, 1] / safe_depth

        ix = px.astype(int)
        iy = py.astype(int)

        in_bounds = valid_depth & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        # Check the mask pixel value for every in-bounds point
        in_mask = np.zeros(len(points_3d), dtype=bool)
        idx = np.where(in_bounds)[0]
        if len(idx):
            in_mask[idx] = mask[iy[idx], ix[idx]] > 0

        keep |= in_mask  # keep if inside mask in ANY frame

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