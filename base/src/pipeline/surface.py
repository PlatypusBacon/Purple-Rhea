"""
Surface Reconstruction — converts the filtered point cloud into a watertight,
3D-printable mesh.

Strategy
--------
1. Reconstruct the occupancy voxel grid from surviving point positions
2. Run marching cubes to extract an isosurface
3. Convert to Blender coordinate system
4. Write .obj file
5. Printability check via open3d

FIX: voxel grid constants now read from config to stay consistent with
     visual_hull.py (original hardcoded r=0.07, z_hi=0.12 which contradicted
     config.VOXEL_XY_EXTENT=0.12, VOXEL_Z_MAX=0.22).

Requires: open3d >= 0.17, scikit-image
    pip install open3d scikit-image
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

    print(f"\n[surface] Input: {len(points_3d)} points")

    if len(points_3d) < 100:
        print(f"    WARNING: only {len(points_3d)} points — mesh will be poor. "
              f"Check visual hull output.")
        if len(points_3d) == 0:
            print("    ERROR: zero points — cannot reconstruct surface. Aborting.")
            return obj_path

    # ------------------------------------------------------------------ #
    # 1. Reconstruct occupancy grid — must match visual_hull.py exactly   #
    # FIX: use config values, not hardcoded constants                     #
    # ------------------------------------------------------------------ #
    r    = config.VOXEL_XY_EXTENT   # was hardcoded 0.07
    z_lo = config.VOXEL_Z_MIN       # was hardcoded -0.02
    z_hi = config.VOXEL_Z_MAX       # was hardcoded 0.12
    res  = 80   # must match grid_resolution in compute_visual_hull

    print(f"[surface] Grid: XY=±{r:.3f}m  Z=[{z_lo:.3f},{z_hi:.3f}]m  res={res}")

    coords_xy = np.linspace(-r,   r,   res)
    coords_z  = np.linspace(z_lo, z_hi, res)

    # Build occupancy grid
    grid = np.zeros((res, res, res), dtype=np.uint8)

    step_xy = coords_xy[1] - coords_xy[0]
    step_z  = coords_z[1]  - coords_z[0]

    ix = np.round((points_3d[:, 0] - coords_xy[0]) / step_xy).astype(int)
    iy = np.round((points_3d[:, 1] - coords_xy[0]) / step_xy).astype(int)
    iz = np.round((points_3d[:, 2] - coords_z[0])  / step_z).astype(int)

    valid = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) & (iz >= 0) & (iz < res)
    n_mapped = int(valid.sum())
    n_outside = int((~valid).sum())
    grid[ix[valid], iy[valid], iz[valid]] = 1

    print(f"[surface] Points mapped to grid: {n_mapped} / {len(points_3d)}")
    if n_outside > 0:
        print(f"[surface] WARNING: {n_outside} points fell outside grid bounds! "
              f"Check VOXEL_XY_EXTENT / VOXEL_Z_MIN / VOXEL_Z_MAX.")
    print(f"[surface] Occupied voxels: {grid.sum()} / {res**3}")

    if grid.sum() == 0:
        print("[surface] ERROR: occupancy grid is all zeros — nothing to mesh.")
        return obj_path

    # ------------------------------------------------------------------ #
    # 2. Marching cubes on the occupancy grid                             #
    # Pad with 1 empty voxel on every face — forces closed surface        #
    # ------------------------------------------------------------------ #
    grid_padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    verts_idx, faces, normals, _ = marching_cubes(grid_padded, level=0.5)

    # Undo the padding offset, then convert voxel indices → world coords
    verts_idx = verts_idx - 1

    verts = np.zeros_like(verts_idx, dtype=np.float64)
    verts[:, 0] = coords_xy[0] + verts_idx[:, 0] * step_xy
    verts[:, 1] = coords_xy[0] + verts_idx[:, 1] * step_xy
    verts[:, 2] = coords_z[0]  + verts_idx[:, 2] * step_z

    print(f"[surface] Marching cubes: {len(verts)} verts, {len(faces)} faces")

    # ------------------------------------------------------------------ #
    # 3. Convert to Blender coordinate system (matches export.py)         #
    #    Blender Y = -world Z,  Blender Z = -world Y                      #
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
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"[surface] Wrote {mesh_path}")

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
#  Camera centres (utility for Poisson normal orientation)                    #
# --------------------------------------------------------------------------- #

def _extract_camera_centres(
    projections: list[np.ndarray] | None,
    points_3d: np.ndarray,
) -> list:
    """
    Extract optical centres from 3×4 projection matrices via SVD null space.
    Falls back to a synthetic point above the cloud if projections unavailable.
    """
    if projections is not None and len(projections) > 0:
        centres = []
        for i, P in enumerate(projections):
            _, _, Vt = np.linalg.svd(P)
            C = Vt[-1]
            if abs(C[3]) < 1e-10:
                print(f"  [surface] WARNING: frame {i} projection null space degenerate")
                continue
            C = C[:3] / C[3]
            centres.append(C.tolist())
            print(f"  [surface] frame {i:02d} camera centre: {C.round(3)}")
        return centres

    centroid = points_3d.mean(axis=0)
    span     = points_3d.max(axis=0) - points_3d.min(axis=0)
    fallback = centroid + np.array([0.0, 0.0, span[2] * 3.0])
    print("    WARNING: no projections supplied — using synthetic camera centre")
    return [fallback.tolist()]


def _orient_normals_multi_view(pcd, camera_centres: list) -> None:
    """Orient each point's normal toward its nearest camera centre."""
    import open3d as o3d
    points  = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    cams    = np.array(camera_centres)
    diff    = points[:, None, :] - cams[None, :, :]
    nearest = (diff ** 2).sum(axis=2).argmin(axis=1)
    to_cam  = cams[nearest] - points
    dot     = (normals * to_cam).sum(axis=1)
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
        print("    TIP: pip install pymeshfix  or open .obj in MeshLab → Filters → Repair")
    else:
        print("    Mesh is ready for 3D printing ✓")
    print(f"    ─────────────────────────────────────────────\n")