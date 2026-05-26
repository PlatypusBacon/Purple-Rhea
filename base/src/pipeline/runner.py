"""
Orchestrates the full reconstruction pipeline.
Each step will be fleshed out in its own module.
"""
from storage.scan_session import ScanSession
import os
import config
import numpy as np
def run(session: ScanSession, on_progress=None) -> str:
    def _progress(stage):
        print(stage)
        if on_progress:
            on_progress(stage)

    if not session.is_complete():
        missing = session.missing_indices()
        raise ValueError(f"Session incomplete — missing frames: {missing}")

    _progress("Starting pipeline")

    # Step 1: Load images
    images = []
    for frame in session:
        img = frame.load_image()
        if img is None:
            raise RuntimeError(f"Could not decode image for frame {frame.index}")
        images.append(img)
    _progress("[1/5] Loaded images")

    # Step 2: Pose computation (servo angles only — no features needed)
    from pipeline.pose_computation import compute_projections
    projections = compute_projections(list(session.frames))
    for i, P in enumerate(projections):
        U, S, Vt = np.linalg.svd(P)
        C = Vt[-1, :3] / Vt[-1, 3]
        print(f"frame {i:02d}: camera centre = {C.round(3)}")
    _progress("[2/5] Projections computed")

    # Step 3: Visual hull
    from pipeline.visual_hull import compute_visual_hull
    points_3d = compute_visual_hull(images, projections, grid_resolution=80)
    _progress(f"[3/5] Visual hull: {len(points_3d)} points")

    # Save point cloud
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(config.OUTPUT_DIR, "reconstruction_raw.npy"), points_3d)
    np.save(os.path.join(config.OUTPUT_DIR, "projections.npy"), np.array(projections))

    # Export .obj
    obj_path = os.path.join(config.OUTPUT_DIR, "reconstruction.obj")
    from pipeline.export import write_obj
    write_obj(points_3d, obj_path)
    _progress("[4/5] Point cloud exported")

    # Surface reconstruction
    if config.RUN_SURFACE_RECON:
        from pipeline.surface import reconstruct_surface
        mesh_path = reconstruct_surface(points_3d, obj_path, projections)
        _progress("[5/5] Surface reconstruction done")
        print(f"  Mesh: {mesh_path}")

    return obj_path