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

    # Debug: show which keypoints were detected and whether they're inside masks
    if getattr(config, "DEBUG_VIZ", True):
        from pipeline.debug_viz import save_keypoint_images, load_masks
        masks = load_masks(len(images))
        save_keypoint_images(images, keypoints_per_frame, masks)

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
    projections = compute_projections(list(session.frames))

    # Step 6: Triangulation → raw point cloud
    from pipeline.triangulation import triangulate
    points_3d = triangulate(matches, projections, keypoints_per_frame)
    print(f"[6/6] Triangulated {len(points_3d)} points")

    # Debug: reproject raw triangulated points onto every frame so you can
    # immediately see if any projection matrix is badly scaled or rotated
    if getattr(config, "DEBUG_VIZ", True):
        from pipeline.debug_viz import save_reprojection_images, load_masks
        masks = load_masks(len(images))
        save_reprojection_images(
            images, points_3d, projections, masks, label="raw"
        )

    # Save raw points for visualiser comparison (before filtering)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(config.OUTPUT_DIR, "reconstruction_raw.npy")
    np.save(raw_path, points_3d)

    # Step 7: Error filtering
    from pipeline.error_filtering import filter_points
    points_3d = filter_points(points_3d, projections, keypoints_per_frame)
    print(f"[7/7] Filtered to {len(points_3d)} points")

    # Debug: reproject filtered points — compare with raw to see what was cut
    if getattr(config, "DEBUG_VIZ", True):
        from pipeline.debug_viz import save_reprojection_images, load_masks
        masks = load_masks(len(images))
        save_reprojection_images(
            images, points_3d, projections, masks, label="filtered"
        )

    # Save filtered points for visualiser
    filtered_path = os.path.join(config.OUTPUT_DIR, "reconstruction_filtered.npy")
    np.save(filtered_path, points_3d)

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