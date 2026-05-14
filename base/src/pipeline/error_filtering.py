"""
Error Filtering — removes outlier 3-D points by checking geometric
consistency across all projection matrices.

Two passes:
  1. Reprojection error filter: reproject each point into every camera and
     discard points whose mean reprojection error exceeds a pixel threshold.
  2. Statistical outlier removal: discard points more than N standard
     deviations from the centroid (catches stray triangulation artefacts).

Input:
    points_3d  : np.ndarray (N, 3)     from triangulation
    projections: list[np.ndarray (3,4)] from pose_computation
    keypoints_per_frame: kept in signature for future per-point track lookup

Output:
    filtered: np.ndarray (M, 3)  where M <= N
"""

import numpy as np

# Pixels — points with mean reprojection error above this are removed
REPROJECTION_THRESHOLD_PX = 8.0

# Statistical filter: remove points beyond this many std-devs from centroid
STATISTICAL_STD_MULTIPLIER = 2.0 #lower is tighter


def filter_points(
    points_3d: np.ndarray,
    projections: list[np.ndarray],
    keypoints_per_frame,
) -> np.ndarray:
    if len(points_3d) == 0:
        return points_3d

    # Pass 1: depth filter
    before = len(points_3d)
    #points_3d = _depth_filter(points_3d, projections)
    #print(f"    depth filter:        {before} -> {len(points_3d)} points")

    # Pass 2: orbital bounds filter — remove points outside the camera circle
    before = len(points_3d)
    points_3d = _orbital_bounds_filter(points_3d)
    print(f"    orbital bounds filter: {before} -> {len(points_3d)} points")

    # Pass 3: statistical outlier removal
    before = len(points_3d)
    points_3d = _statistical_filter(points_3d)
    print(f"    statistical filter:  {before} -> {len(points_3d)} points")

    return points_3d


# --------------------------------------------------------------------------- #
#  Pass 1 — reprojection                                                       #
# --------------------------------------------------------------------------- #

def _reprojection_filter(
    points_3d: np.ndarray,
    projections: list[np.ndarray],
) -> np.ndarray:
    """
    For each 3-D point X, reproject into every camera P_i:
        x_proj = P_i @ [X; 1]   (homogeneous)
    Average the reprojection error across all cameras and keep points below
    REPROJECTION_THRESHOLD_PX.

    Points behind a camera (negative depth) are also discarded.
    """
    N = len(points_3d)
    X_h = np.hstack([points_3d, np.ones((N, 1))])   # (N, 4) homogeneous

    errors = np.zeros(N, dtype=np.float64)
    behind_count = np.zeros(N, dtype=int)

    for P in projections:
        proj = (P @ X_h.T).T          # (N, 3)
        depths = proj[:, 2]
        behind_count += (depths <= 0).astype(int)

        # Normalise to pixel coords
        px = proj[:, 0] / np.where(depths != 0, depths, 1e-9)
        py = proj[:, 1] / np.where(depths != 0, depths, 1e-9)

        # We don't have a per-point ground-truth pixel here, so we use
        # the reprojection spread across cameras as a proxy for consistency.
        # A more precise filter is possible once point tracks are stored.
        errors += np.sqrt(px**2 + py**2)   # distance from principal ray

    mean_errors = errors / max(len(projections), 1)

    # Keep points that are in front of all cameras and have low error
    mask = (behind_count == 0) & (mean_errors < REPROJECTION_THRESHOLD_PX * 100)
    return points_3d[mask]


# --------------------------------------------------------------------------- #
#  Pass 2 — statistical                                                        #
# --------------------------------------------------------------------------- #

def _statistical_filter(points_3d: np.ndarray) -> np.ndarray:
    """
    Remove points more than STATISTICAL_STD_MULTIPLIER standard deviations
    from the centroid in Euclidean distance.
    """
    if len(points_3d) < 4:
        return points_3d

    centroid = points_3d.mean(axis=0)
    dists = np.linalg.norm(points_3d - centroid, axis=1)
    threshold = dists.mean() + STATISTICAL_STD_MULTIPLIER * dists.std()
    return points_3d[dists < threshold]

def _orbital_bounds_filter(points_3d: np.ndarray) -> np.ndarray:
    centroid = points_3d.mean(axis=0)
    centred = points_3d - centroid

    # PCA to find orbital plane
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    coords_2d = centred @ Vt[:2].T
    radial_dist = np.linalg.norm(coords_2d, axis=1)

    # Use data-inferred radius only — SfM scale is arbitrary,
    # NOMINAL_RADIUS in metres is meaningless here
    inferred_radius = np.percentile(radial_dist, 90)
    MARGIN = 0.75   # keep inner 75% — cuts background, keeps object
    cutoff = inferred_radius * MARGIN

    print(f"      inferred radius: {inferred_radius:.4f}  cutoff: {cutoff:.4f}  "
          f"({(radial_dist < cutoff).sum()} / {len(points_3d)} kept)")
    return points_3d[radial_dist < cutoff]

def _depth_filter(
    points_3d: np.ndarray,
    projections: list[np.ndarray],
) -> np.ndarray:
    """
    Discard any point that projects behind any camera (depth <= 0).
    These are degenerate triangulations from near-parallel viewing rays.
    """
    N = len(points_3d)
    X_h = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    keep = np.ones(N, dtype=bool)

    for P in projections:
        proj = (P @ X_h.T).T        # (N, 3)
        depths = proj[:, 2]
        keep &= (depths > 0)

    return points_3d[keep]