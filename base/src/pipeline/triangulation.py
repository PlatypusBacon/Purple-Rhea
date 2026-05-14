"""
Triangulation — recovers 3-D point positions from matched 2-D correspondences
using the Direct Linear Transform (DLT).

For each matched pair (i, j) and each good DMatch:
  - Look up the 2-D pixel coordinates of the matched keypoints
  - Stack the projection equations from P_i and P_j
  - Solve via SVD to get the homogeneous 3-D point X
  - Divide by w to get Euclidean (x, y, z)

Input:
    matches           : dict {(i,j): [DMatch]}         from feature_matching
    projections       : list[np.ndarray shape (3,4)]    from pose_computation
    keypoints_per_frame: list[list[cv2.KeyPoint]]       from feature_detection

Output:
    points_3d: np.ndarray shape (N, 3)  — raw point cloud before filtering
"""

import numpy as np
import cv2


def triangulate(
    matches: dict[tuple[int, int], list[cv2.DMatch]],
    projections: list[np.ndarray],
    keypoints_per_frame: list[list[cv2.KeyPoint]],
) -> np.ndarray:
    """
    Triangulate all matched pairs into a single point cloud.
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

    return np.vstack(all_points)


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

    For each point pair (x1, x2):
      Build A (4x4) from the cross-product form of the projection equations:
        x1 x (P1 X) = 0
        x2 x (P2 X) = 0
      Solve via SVD: X = last right singular vector, then dehomogenise.

    Uses OpenCV's triangulatePoints as the DLT solver (it uses the same
    formulation and is numerically well-tested).

    pts1, pts2: (M, 2) pixel coordinates
    Returns:    (M, 3) Euclidean 3-D points
    """
    # cv2.triangulatePoints expects (2, M) float32
    p1 = pts1.T.astype(np.float32)
    p2 = pts2.T.astype(np.float32)

    X_hom = cv2.triangulatePoints(
        P1.astype(np.float32),
        P2.astype(np.float32),
        p1, p2,
    )  # (4, M)

    # Dehomogenise: divide x, y, z by w
    X_hom /= X_hom[3:4, :]
    return X_hom[:3, :].T   # (M, 3)