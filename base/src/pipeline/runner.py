"""
Orchestrates the full reconstruction pipeline.
Each step will be fleshed out in its own module.
"""
from storage.scan_session import ScanSession
import os
import config
import numpy as np
def run(session: ScanSession) -> str:
    if not session.is_complete():
        missing = session.missing_indices()
        raise ValueError(f"Session incomplete — missing frames: {missing}")

    print(f"\n{'='*60}")
    print(f"  Starting pipeline — {len(session)} frames")
    print(f"{'='*60}\n")

    # Step 1: Load images
    images = []
    for frame in session:
        img = frame.load_image()
        if img is None:
            raise RuntimeError(f"Could not decode image for frame {frame.index}")
        images.append(img)
    print(f"[1/3] Loaded {len(images)} images")

    # Step 2: Pose computation (servo angles only — no features needed)
    from pipeline.pose_computation import compute_projections
    projections = compute_projections(list(session.frames))
    for i, P in enumerate(projections):
        U, S, Vt = np.linalg.svd(P)
        C = Vt[-1, :3] / Vt[-1, 3]
        print(f"frame {i:02d}: camera centre = {C.round(3)}")
    print(f"[2/3] Projections computed")

    # Step 3: Visual hull
    from pipeline.visual_hull import compute_visual_hull
    points_3d = compute_visual_hull(images, projections, grid_resolution=80)
    print(f"[3/3] Visual hull: {len(points_3d)} points")

    # Save point cloud
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(config.OUTPUT_DIR, "reconstruction_raw.npy"), points_3d)
    np.save(os.path.join(config.OUTPUT_DIR, "projections.npy"), np.array(projections))

    # Export .obj
    obj_path = os.path.join(config.OUTPUT_DIR, "reconstruction.obj")
    from pipeline.export import write_obj
    write_obj(points_3d, obj_path)

    # Surface reconstruction
    if config.RUN_SURFACE_RECON:
        from pipeline.surface import reconstruct_surface
        mesh_path = reconstruct_surface(points_3d, obj_path, projections)
        print(f"  Mesh: {mesh_path}")

    return obj_path