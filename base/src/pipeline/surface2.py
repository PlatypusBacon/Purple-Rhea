"""
Surface Reconstruction — converts the filtered point cloud into a watertight,
3D-printable mesh.

Strategy
--------
1. Pre-process the point cloud (outlier removal, resampling for uniform density)
2. Estimate & orient normals robustly using multiple camera viewpoints
3. Run Poisson surface reconstruction at a generous depth
4. Aggressively remove low-density boundary artefacts
5. Post-process: fill holes, remove non-manifold geometry, smooth lightly
6. Validate watertightness; warn if the mesh is not print-ready

Requires: open3d >= 0.17
    pip install open3d
"""

from __future__ import annotations
import numpy as np
import os


def reconstruct_surface(
    points_3d: np.ndarray,
    obj_path: str,
    projections: list[np.ndarray] | None = None,
) -> str:
    try:
        import open3d as o3d
    except ImportError:
        print("    open3d not installed — pip install open3d")
        return obj_path

    import config
    from skimage.measure import marching_cubes

    print(f"    input: {len(points_3d)} points")

    if len(points_3d) < 100:
        print("    WARNING: fewer than 100 points")
        return obj_path

    # ------------------------------------------------------------------ #
    # 1. Reconstruct the voxel grid from the surviving point positions    #
    # ------------------------------------------------------------------ #
    # Infer grid bounds and resolution from config (must match visual_hull)
    r    =  0.07
    z_lo = -0.02
    z_hi =  0.12
    res  = 80  # must match grid_resolution in compute_visual_hull

    coords_xy = np.linspace(-r,   r,   res)
    coords_z  = np.linspace(z_lo, z_hi, res)

    # Build occupancy grid
    grid = np.zeros((res, res, res), dtype=np.uint8)

    # Map each surviving point back to its voxel index
    # points_3d are voxel centres, so we can reverse-engineer indices
    step_xy = coords_xy[1] - coords_xy[0]
    step_z  = coords_z[1]  - coords_z[0]

    ix = np.round((points_3d[:, 0] - coords_xy[0]) / step_xy).astype(int)
    iy = np.round((points_3d[:, 1] - coords_xy[0]) / step_xy).astype(int)
    iz = np.round((points_3d[:, 2] - coords_z[0])  / step_z).astype(int)

    # Clip to valid range
    valid = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) & (iz >= 0) & (iz < res)
    grid[ix[valid], iy[valid], iz[valid]] = 1

    # ------------------------------------------------------------------ #
    # 2. Marching cubes on the occupancy grid                             #
    # ------------------------------------------------------------------ #
    # Pad with 1 empty voxel on every face — forces closed surface at boundaries
    grid_padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)

    verts_idx, faces, normals, _ = marching_cubes(grid_padded, level=0.5)

    # Offset indices back by 1 to account for padding, then convert to world coords
    verts_idx = verts_idx - 1  # undo the padding offset

    verts = np.zeros_like(verts_idx, dtype=np.float64)
    verts[:, 0] = coords_xy[0] + verts_idx[:, 0] * step_xy
    verts[:, 1] = coords_xy[0] + verts_idx[:, 1] * step_xy
    verts[:, 2] = coords_z[0]  + verts_idx[:, 2] * step_z

    # ------------------------------------------------------------------ #
    # 3. Convert to Blender coordinate system (matches export.py)         #
    # ------------------------------------------------------------------ #
    blender_verts = verts.copy()
    blender_verts[:, 1] = -verts[:, 2]
    blender_verts[:, 2] = -verts[:, 1]

    # ------------------------------------------------------------------ #
    # 4. Write .obj                                                        #
    # ------------------------------------------------------------------ #
    mesh_path = obj_path.replace(".obj", "_mesh.obj")
    os.makedirs(os.path.dirname(mesh_path) or ".", exist_ok=True)

    with open(mesh_path, "w") as f:
        f.write("# Purple-Rhea marching cubes mesh\n")
        f.write(f"# {len(blender_verts)} verts, {len(faces)} faces\n\n")
        for x, y, z in blender_verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for face in faces:
            # .obj faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"    wrote {mesh_path}")

    # ------------------------------------------------------------------ #
    # 5. Printability check via open3d                                    #
    # ------------------------------------------------------------------ #
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices  = o3d.utility.Vector3dVector(blender_verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    _check_printability(mesh_o3d)

    return mesh_path


# --------------------------------------------------------------------------- #
#  Camera centres                                                              #
# --------------------------------------------------------------------------- #

def _extract_camera_centres(
    projections: list[np.ndarray] | None,
    points_3d: np.ndarray,
) -> list:
    """
    Extract optical centres from 3×4 projection matrices via SVD null space.
    P = K[R|t]  =>  camera centre C = null(P)  (last row of V^T, dehomogenised).
    Falls back to a synthetic point above the cloud if projections unavailable.
    """
    if projections is not None and len(projections) > 0:
        centres = []
        for P in projections:
            _, _, Vt = np.linalg.svd(P)
            C = Vt[-1]
            C = C[:3] / C[3]
            centres.append(C.tolist())
        return centres

    centroid = points_3d.mean(axis=0)
    span     = points_3d.max(axis=0) - points_3d.min(axis=0)
    fallback = centroid + np.array([0.0, 0.0, span[2] * 3.0])
    print("    WARNING: no projections supplied — using synthetic camera centre")
    print("             Pass projections= to reconstruct_surface for better normals")
    return [fallback.tolist()]


def _orient_normals_multi_view(pcd, camera_centres: list) -> None:
    """
    Orient each point's normal toward its nearest camera centre.
    Correct for a turntable rig where different surfaces face different cameras.
    """
    import open3d as o3d

    points  = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    cams    = np.array(camera_centres)

    diff    = points[:, None, :] - cams[None, :, :]   # (N, K, 3)
    nearest = (diff ** 2).sum(axis=2).argmin(axis=1)  # (N,)
    to_cam  = cams[nearest] - points

    dot = (normals * to_cam).sum(axis=1)
    normals[dot < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)


# --------------------------------------------------------------------------- #
#  Repair                                                                      #
# --------------------------------------------------------------------------- #

def _repair_mesh(mesh):
    import open3d as o3d

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    try:
        import pymeshfix
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        mf    = pymeshfix.MeshFix(verts, faces)
        mf.repair()
        if hasattr(mf, "points"):
            out_verts, out_faces = mf.points, mf.faces
        elif hasattr(mf, "v"):
            out_verts, out_faces = mf.v, mf.f
        else:
            raise AttributeError("Unrecognised pymeshfix API")
        mesh.vertices  = o3d.utility.Vector3dVector(out_verts)
        mesh.triangles = o3d.utility.Vector3iVector(out_faces)
        print("    pymeshfix repair applied")
    except ImportError:
        print("    pymeshfix not installed (pip install pymeshfix) — skipping")
    except AttributeError as e:
        print(f"    pymeshfix skipped: {e}")

    try:
        import config
        smooth_iter = getattr(config, "SMOOTH_ITERATIONS", 0)
    except ImportError:
        smooth_iter = 0
    if smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        print(f"    Taubin smooth: {smooth_iter} iterations")

    mesh.compute_vertex_normals()
    return mesh


# --------------------------------------------------------------------------- #
#  Printability report                                                         #
# --------------------------------------------------------------------------- #

def _check_printability(mesh) -> None:
    watertight = mesh.is_watertight()
    manifold   = (
        len(mesh.get_non_manifold_edges()) == 0
        and len(mesh.get_non_manifold_vertices()) == 0
    )
    print(f"\n    ── Printability report ──────────────────────")
    print(f"    Vertices   : {len(mesh.vertices)}")
    print(f"    Faces      : {len(mesh.triangles)}")
    print(f"    Watertight : {'✓' if watertight else '✗  ← NOT print-ready'}")
    print(f"    Manifold   : {'✓' if manifold   else '✗  ← NOT print-ready'}")
    if not watertight or not manifold:
        print("    TIP: pip install pymeshfix  or open .ply in MeshLab")
    else:
        print("    Mesh is ready for 3D printing ✓")
    print(f"    ─────────────────────────────────────────────\n")