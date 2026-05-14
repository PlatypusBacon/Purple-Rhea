"""
Surface Reconstruction — converts the filtered point cloud into a mesh.

Two methods available via config.SURFACE_METHOD:
  'ball_pivot'  : Ball-Pivoting Algorithm — fast, good for dense uniform clouds,
                  preserves sharp edges (good for the metal box test object)
  'poisson'     : Poisson surface reconstruction — smoother, watertight mesh,
                  requires normals (estimated here via PCA on neighbours)

Requires open3d:
    pip install open3d
"""

import numpy as np
import os


def reconstruct_surface(points_3d: np.ndarray, obj_path: str) -> str:
    """
    Build a mesh from the point cloud and write it to obj_path.
    Returns the output path.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("    open3d not installed — writing point cloud only")
        print("    Install with: pip install open3d")
        return obj_path

    import config
    method = getattr(config, "SURFACE_METHOD", "ball_pivot")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Estimate normals — required by both methods
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    if method == "poisson":
        mesh = _poisson(pcd)
    else:
        mesh = _ball_pivot(pcd, points_3d)

    if mesh is None:
        print("    Surface reconstruction failed — no mesh produced")
        return obj_path

    # Clean up mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    print(f"    mesh: {n_verts} vertices, {n_faces} faces")

    # Write as .obj
    mesh_path = obj_path.replace(".obj", "_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=True)
    print(f"    wrote mesh to {mesh_path}")

    # Also write a .ply for easier inspection in MeshLab/Blender
    ply_path = obj_path.replace(".obj", "_mesh.ply")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    print(f"    wrote mesh to {ply_path}")

    return mesh_path


# --------------------------------------------------------------------------- #
#  Methods                                                                     #
# --------------------------------------------------------------------------- #

def _ball_pivot(pcd, points_3d: np.ndarray):
    """
    Ball-Pivoting Algorithm.
    Ball radii are estimated from the average nearest-neighbour distance.
    """
    import open3d as o3d

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist  = np.mean(distances)

    # Use three ball sizes: tight, medium, loose
    radii = [avg_dist * 0.5, avg_dist, avg_dist * 2.0, avg_dist * 4.0]
    print(f"    ball pivot radii: {[f'{r:.4f}' for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh


def _poisson(pcd):
    """
    Poisson surface reconstruction.
    depth controls resolution — higher = finer but slower.
    """
    import open3d as o3d

    depth = 8   # increase to 9-10 for finer detail on the actual rig
    print(f"    poisson depth={depth}")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Remove low-density vertices (boundary artefacts)
    densities_np = np.asarray(densities)
    threshold = np.percentile(densities_np, 5)
    verts_to_remove = densities_np < threshold
    mesh.remove_vertices_by_mask(verts_to_remove)

    return mesh