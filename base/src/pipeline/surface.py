"""
Surface reconstruction from point clouds.

This module expects a point cloud (Nx3) and creates a closed mesh suitable for
3D printing workflows:
  1) point cleanup (dedupe + outlier filtering + largest cluster)
  2) normal estimation/orientation
  3) surface reconstruction from configurable methods
  4) mesh repair + manifold/watertight checks
"""

from __future__ import annotations

import os
import numpy as np


def reconstruct_surface(
    points_3d: np.ndarray,
    obj_path: str,
    projections: list[np.ndarray] | None = None,
) -> str:
    try:
        import open3d as o3d
    except ImportError:
        print("    open3d not installed - pip install open3d")
        return obj_path

    import config

    pts = _coerce_points(points_3d)
    print(f"\n[surface] Input points: {len(pts)}")
    if len(pts) < 50:
        print("    too few points for reliable meshing")
        return obj_path

    pcd, prep = _prepare_point_cloud(pts, o3d)
    if pcd is None or len(pcd.points) < 50:
        print("    point cleaning removed too much data - aborting mesh reconstruction")
        return obj_path

    print(
        f"[surface] Cleaned points: {len(pcd.points)} "
        f"(voxel={prep['voxel_size']:.6f}m, nn={prep['median_nn']:.6f}m)"
    )

    camera_centres = _extract_camera_centres(projections, np.asarray(pcd.points))
    methods = _resolve_surface_methods(config)
    print(f"[surface] Method order: {methods}")

    mesh = None
    normals_ready = False

    for method in methods:
        method = str(method).strip().lower()
        if not method:
            continue

        if _method_needs_normals(method) and not normals_ready:
            _estimate_normals(pcd, o3d, prep, camera_centres)
            normals_ready = True

        try:
            candidate = _reconstruct_with_method(method, pcd, o3d, prep, config)
        except Exception as exc:
            print(f"[surface] method '{method}' failed: {exc}")
            continue

        if candidate is None or len(candidate.vertices) == 0 or len(candidate.triangles) == 0:
            print(f"[surface] method '{method}' produced an empty mesh")
            continue

        candidate = _keep_largest_mesh_component(candidate)
        candidate = _simplify_mesh(candidate, config)
        candidate = _repair_mesh(candidate)

        mesh = candidate
        print(
            f"[surface] method '{method}' succeeded: "
            f"{len(mesh.vertices)} verts, {len(mesh.triangles)} faces"
        )
        if mesh.is_watertight():
            break

    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[surface] ERROR: all surface methods failed")
        return obj_path

    if not mesh.is_watertight():
        print("[surface] Mesh is not watertight after first repair - retrying Poisson fallback")
        mesh_fallback = _mesh_poisson(pcd, o3d, config, depth_boost=1)
        if mesh_fallback is not None and len(mesh_fallback.triangles) > 0:
            mesh_fallback = _keep_largest_mesh_component(mesh_fallback)
            mesh_fallback = _repair_mesh(mesh_fallback)
            if mesh_fallback.is_watertight() or len(mesh_fallback.triangles) > len(mesh.triangles):
                mesh = mesh_fallback

    mesh.compute_vertex_normals()

    mesh_path = obj_path.replace(".obj", "_mesh.obj")
    os.makedirs(os.path.dirname(mesh_path) or ".", exist_ok=True)
    _write_mesh_obj_blender_coords(mesh, mesh_path)

    print(f"[surface] Wrote {mesh_path}")
    _check_printability(mesh)
    return mesh_path


def _resolve_surface_methods(config) -> list[str]:
    raw = getattr(config, "SURFACE_METHODS", None)
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        if parts:
            return parts
    if isinstance(raw, (list, tuple)):
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
        if parts:
            return parts

    default_method = str(getattr(config, "SURFACE_METHOD", "convex_hull")).strip().lower()
    return [default_method] if default_method else ["convex_hull"]


def _method_needs_normals(method: str) -> bool:
    return method in {"poisson", "screened_poisson", "ball_pivoting", "bpa", "ball"}


def _reconstruct_with_method(method: str, pcd, o3d, prep: dict, config):
    if method in {"convex_hull", "hull"}:
        return _mesh_convex_hull(pcd, config)
    if method in {"poisson", "screened_poisson"}:
        return _mesh_poisson(pcd, o3d, config)
    if method in {"ball_pivoting", "bpa", "ball"}:
        return _mesh_ball_pivoting(pcd, o3d, prep)
    if method in {"alpha", "alpha_shape"}:
        return _mesh_alpha_shape(pcd, o3d, prep)
    print(f"[surface] unknown method '{method}' - skipping")
    return None


# --------------------------------------------------------------------------- #
# Point cloud preparation
# --------------------------------------------------------------------------- #


def _coerce_points(points_3d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N,3), got {pts.shape}")
    finite = np.isfinite(pts).all(axis=1)
    if not finite.all():
        pts = pts[finite]
        print(f"    removed {int((~finite).sum())} non-finite points")
    return pts


def _prepare_point_cloud(pts: np.ndarray, o3d):
    import config

    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 50:
        return None, {}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    diag = float(np.linalg.norm(extent))
    voxel_size = getattr(config, "VOXEL_SIZE", None)
    if voxel_size is None or voxel_size <= 0:
        voxel_size = max(diag / 220.0, 2e-4)

    pcd = pcd.voxel_down_sample(float(voxel_size))

    nb_neighbors = int(max(10, getattr(config, "OUTLIER_NB_NEIGHBORS", 20)))
    std_ratio = float(max(0.5, getattr(config, "OUTLIER_STD_RATIO", 2.0)))
    if len(pcd.points) > nb_neighbors * 2:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    nn = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    median_nn = float(np.median(nn)) if nn.size > 0 else float(voxel_size)

    radius = max(median_nn * 4.0, voxel_size * 3.0)
    min_points = max(5, min(20, int(round(len(pcd.points) * 0.005))))
    if len(pcd.points) > min_points * 2:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=min_points,
            radius=float(radius),
        )

    pcd = _keep_largest_point_cluster(pcd, eps=max(median_nn * 5.0, voxel_size * 4.0))

    prep = {
        "voxel_size": float(voxel_size),
        "median_nn": float(median_nn),
        "normal_radius": float(max(median_nn * 4.0, voxel_size * 3.0)),
        "normal_max_nn": int(max(30, min(100, round(len(pcd.points) * 0.03)))),
    }
    return pcd, prep


def _keep_largest_point_cluster(pcd, eps: float):
    n = len(pcd.points)
    if n < 80:
        return pcd

    min_points = max(15, min(120, int(round(n * 0.02))))
    labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=min_points, print_progress=False))

    valid = labels >= 0
    if not np.any(valid):
        print("    DBSCAN found no dense cluster - keeping all points")
        return pcd

    uniq, counts = np.unique(labels[valid], return_counts=True)
    keep_label = int(uniq[np.argmax(counts)])
    keep_idx = np.where(labels == keep_label)[0]
    removed = n - len(keep_idx)
    if removed > 0:
        print(f"    removed {removed} points from small/disconnected clusters")

    return pcd.select_by_index(keep_idx)


# --------------------------------------------------------------------------- #
# Normal estimation and reconstruction methods
# --------------------------------------------------------------------------- #


def _estimate_normals(pcd, o3d, prep: dict, camera_centres: list[list[float]]) -> None:
    radius = float(prep["normal_radius"])
    max_nn = int(prep["normal_max_nn"])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    _orient_normals_multi_view(pcd, camera_centres)


def _mesh_poisson(pcd, o3d, config, depth_boost: int = 0):
    depth = int(getattr(config, "POISSON_DEPTH", 9)) + int(depth_boost)
    scale = float(getattr(config, "POISSON_SCALE", 1.1))
    linear_fit = bool(getattr(config, "POISSON_LINEAR_FIT", False))
    keep_pct = float(getattr(config, "POISSON_DENSITY_KEEP_PERCENTILE", 15))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=max(6, depth),
        scale=max(1.01, scale),
        linear_fit=linear_fit,
    )

    densities = np.asarray(densities)
    if densities.size > 0:
        keep_pct = min(max(keep_pct, 0.0), 60.0)
        cutoff = np.percentile(densities, keep_pct)
        remove = densities < cutoff
        if np.any(remove):
            mesh.remove_vertices_by_mask(remove)
            print(f"    Poisson density trim: removed {int(remove.sum())} low-density verts")

    return mesh


def _mesh_convex_hull(pcd, config):
    joggle = bool(getattr(config, "CONVEX_HULL_JOGGLE_INPUTS", True))
    try:
        mesh, _ = pcd.compute_convex_hull(joggle_inputs=joggle)
    except TypeError:
        mesh, _ = pcd.compute_convex_hull()

    mesh.compute_vertex_normals()
    return mesh


def _mesh_ball_pivoting(pcd, o3d, prep: dict):
    r = float(prep["median_nn"])
    radii = [max(r * 1.5, 1e-4), max(r * 3.0, 2e-4), max(r * 6.0, 4e-4)]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    return mesh


def _mesh_alpha_shape(pcd, o3d, prep: dict):
    alpha = float(max(prep["median_nn"] * 6.0, prep["voxel_size"] * 4.0, 1e-4))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh


def _simplify_mesh(mesh, config):
    target = getattr(config, "SURFACE_TARGET_TRIANGLES", None)
    if target is None:
        return mesh
    try:
        target_i = int(target)
    except (TypeError, ValueError):
        return mesh
    if target_i <= 0 or len(mesh.triangles) <= target_i:
        return mesh
    try:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_i)
        print(f"    simplified mesh to target {target_i} triangles")
    except Exception as exc:
        print(f"    simplify failed: {exc}")
    return mesh


# --------------------------------------------------------------------------- #
# Camera centres and normal orientation
# --------------------------------------------------------------------------- #


def _extract_camera_centres(
    projections: list[np.ndarray] | None,
    points_3d: np.ndarray,
) -> list[list[float]]:
    if projections is not None and len(projections) > 0:
        centres: list[list[float]] = []
        for i, P in enumerate(projections):
            _, _, vt = np.linalg.svd(P)
            c = vt[-1]
            if abs(c[3]) < 1e-10:
                continue
            c = c[:3] / c[3]
            centres.append(c.tolist())
            if i < 6 or i % 6 == 0:
                print(f"  [surface] frame {i:02d} camera centre: {np.round(c, 3)}")
        if centres:
            return centres

    centroid = points_3d.mean(axis=0)
    span = points_3d.max(axis=0) - points_3d.min(axis=0)
    fallback = centroid + np.array([0.0, 0.0, max(span[2] * 3.0, 0.2)])
    print("    no usable camera centres - using synthetic viewpoint")
    return [fallback.tolist()]


def _orient_normals_multi_view(pcd, camera_centres: list[list[float]]) -> None:
    import open3d as o3d

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    cams = np.asarray(camera_centres, dtype=np.float64)

    diff = points[:, None, :] - cams[None, :, :]
    nearest = (diff * diff).sum(axis=2).argmin(axis=1)
    to_cam = cams[nearest] - points
    dot = (normals * to_cam).sum(axis=1)
    normals[dot < 0] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(normals)


# --------------------------------------------------------------------------- #
# Mesh cleanup / export
# --------------------------------------------------------------------------- #


def _keep_largest_mesh_component(mesh):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) == 0:
        return mesh

    tri_clusters = np.asarray(triangle_clusters)
    tri_counts = np.asarray(cluster_n_triangles)
    keep = int(np.argmax(tri_counts))

    remove_mask = tri_clusters != keep
    if np.any(remove_mask):
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
    return mesh


def _repair_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    try:
        import pymeshfix
        import open3d as o3d

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if len(verts) > 0 and len(faces) > 0:
            mf = pymeshfix.MeshFix(verts, faces)
            mf.repair()
            out_verts = mf.points if hasattr(mf, "points") else mf.v
            out_faces = mf.faces if hasattr(mf, "faces") else mf.f
            mesh.vertices = o3d.utility.Vector3dVector(out_verts)
            mesh.triangles = o3d.utility.Vector3iVector(out_faces)
            print("    pymeshfix repair applied")
    except Exception:
        pass

    try:
        import config

        smooth_iter = int(max(0, getattr(config, "SMOOTH_ITERATIONS", 0)))
        if smooth_iter > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
            print(f"    Taubin smooth: {smooth_iter} iterations")
    except Exception:
        pass

    mesh.compute_vertex_normals()
    return mesh


def _write_mesh_obj_blender_coords(mesh, path: str) -> None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    # OpenCV/world -> Blender convention used elsewhere in project:
    #   Blender_X = X, Blender_Y = -Z, Blender_Z = -Y
    bverts = verts.copy()
    bverts[:, 1] = -verts[:, 2]
    bverts[:, 2] = -verts[:, 1]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Purple-Rhea mesh from point cloud\n")
        f.write(f"# {len(bverts)} verts, {len(faces)} faces\n\n")
        for x, y, z in bverts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")


# --------------------------------------------------------------------------- #
# Printability report
# --------------------------------------------------------------------------- #


def _check_printability(mesh) -> None:
    watertight = bool(mesh.is_watertight())
    manifold = (
        len(mesh.get_non_manifold_edges()) == 0
        and len(mesh.get_non_manifold_vertices()) == 0
    )

    print("\n    -- Printability report --")
    print(f"    Vertices   : {len(mesh.vertices)}")
    print(f"    Faces      : {len(mesh.triangles)}")
    print(f"    Watertight : {'yes' if watertight else 'no'}")
    print(f"    Manifold   : {'yes' if manifold else 'no'}")
    if watertight and manifold:
        print("    Mesh is printable-ready")
    else:
        print("    Mesh still needs repair/cleanup for print-ready output")
    print("    --------------------------\n")
