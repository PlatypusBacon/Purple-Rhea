"""
Export — writes the reconstructed point cloud to a Wavefront .obj file.

Vertices only for now.  Normal estimation (PCA on neighbours) is stubbed
and will be filled in when time permits, as noted in the wiki.

Input:
    points_3d: np.ndarray (N, 3)
    path:      output file path  e.g. "output/reconstruction.obj"

Output:
    the .obj file on disk
"""

import numpy as np
import os


def write_obj(points_3d: np.ndarray, path: str) -> None:
    """
    Write vertices (and normals if provided) to a .obj file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    normals = _estimate_normals(points_3d)   # None until implemented

    with open(path, "w") as f:
        f.write("# Purple-Rhea 3D reconstruction\n")
        f.write(f"# {len(points_3d)} vertices\n\n")

        for x, y, z in points_3d:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        if normals is not None:
            f.write("\n")
            for nx, ny, nz in normals:
                f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

    print(f"    wrote {len(points_3d)} vertices to {path}")


# --------------------------------------------------------------------------- #
#  Normal estimation stub                                                      #
# --------------------------------------------------------------------------- #

def _estimate_normals(
    points_3d: np.ndarray,
    k_neighbours: int = 10,
) -> np.ndarray | None:
    """
    PCA-based surface normal estimation.

    For each point, fit a plane to its k nearest neighbours and take the
    smallest eigenvector as the normal.

    Currently returns None (stub).  Enable by changing the return statement
    below to `return normals` when this step is ready.
    """

    if len(points_3d) < k_neighbours + 1:
        return None

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("    normal estimation skipped — install scikit-learn to enable")
        return None

    nbrs = NearestNeighbors(n_neighbors=k_neighbours + 1).fit(points_3d)
    _, indices = nbrs.kneighbors(points_3d)

    normals = np.zeros_like(points_3d)
    for i, neighbours in enumerate(indices):
        neighbourhood = points_3d[neighbours]      # (k+1, 3)
        centred = neighbourhood - neighbourhood.mean(axis=0)
        _, _, Vt = np.linalg.svd(centred)
        normals[i] = Vt[-1]                        # smallest singular vector

    # Orient normals consistently toward centroid
    centroid = points_3d.mean(axis=0)
    for i, (p, n) in enumerate(zip(points_3d, normals)):
        if np.dot(n, centroid - p) > 0:
            normals[i] = -n

    return normals