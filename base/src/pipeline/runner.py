"""
Orchestrates the full reconstruction pipeline.
Each step will be fleshed out in its own module.
"""

from storage.scan_session import ScanSession
import os
import config
import numpy as np

def run(session: ScanSession) -> str:
    """
    Takes a complete ScanSession and runs the full pipeline.
    Returns the path to the exported .obj file.
    """
    if not session.is_complete():
        missing = session.missing_indices()
        raise ValueError(f"Session incomplete — missing frames: {missing}")

    print(f"\n{'='*60}")
    print(f"  Starting pipeline — {len(session)} frames")
    print(f"{'='*60}\n")

    # Load all images into memory once
    images = []
    for frame in session:
        img = frame.load_image()
        if img is None:
            raise RuntimeError(f"Could not decode image for frame {frame.index}")
        images.append(img)
    print(f"[1/6] Loaded {len(images)} images")

    # Step 2: Feature detection (DoG keypoints — see feature_detection.py)
    from pipeline.feature_detection import detect_all
    keypoints_per_frame = detect_all(images)
    print(f"[2/6] Feature detection complete")

    # Step 3: Feature description
    from pipeline.feature_description import describe_all
    descriptors_per_frame = describe_all(images, keypoints_per_frame)
    print(f"[3/6] Feature description complete")

    # Step 4: Feature matching between adjacent (and non-adjacent) frame pairs
    from pipeline.feature_matching import match_all_pairs
    matches = match_all_pairs(descriptors_per_frame)
    print(f"[4/6] Feature matching complete — {len(matches)} pairs matched")

    # Step 5: Pose computation from servo angles + IMU data
    from pipeline.pose_computation import compute_projections
    projections = compute_projections(
        list(session.frames),
        matches=matches if config.DEBUG_MODE else None,
        keypoints_per_frame=keypoints_per_frame if config.DEBUG_MODE else None,
    )

    # Step 6: Triangulation → raw point cloud
    from pipeline.triangulation import triangulate
    points_3d = triangulate(matches, projections, keypoints_per_frame)
    print(f"[6/6] Triangulated {len(points_3d)} points")

    # Step 7: Error filtering
    from pipeline.error_filtering import filter_points
    points_3d = filter_points(points_3d, projections, keypoints_per_frame)
    print(f"[7/7] Filtered to {len(points_3d)} points")

        
    # Define this before the export block
    obj_path = os.path.join(config.OUTPUT_DIR, "reconstruction.obj")

    # Export
    from pipeline.export import write_obj
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    write_obj(points_3d, obj_path)

    # Save projections for visualiser
    proj_path = os.path.join(config.OUTPUT_DIR, "projections.npy")
    np.save(proj_path, np.array(projections))

    # Surface reconstruction
    if config.RUN_SURFACE_RECON:
        from pipeline.surface import reconstruct_surface
        mesh_path = reconstruct_surface(points_3d, obj_path)
        print(f"  Mesh: {mesh_path}")

    return obj_path