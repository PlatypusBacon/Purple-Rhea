"""
Pose Computation — builds a 3x4 projection matrix P = K [R | t] for each
frame, using the known camera intrinsics and the pose from the Kalman filter
/ servo angle.

The camera always points inward toward the origin (the turntable centre), so
the translation vector is simply -R @ position_3d.

Input:
    frames: list[ScanFrame]  — need frame.pose for each

Output:
    projections: list[np.ndarray]  — each is shape (3, 4)
                 projections[i] corresponds to frames[i]
"""

import numpy as np
import math
import config


def compute_projections(frames, matches=None, keypoints_per_frame=None) -> list[np.ndarray]:
    """
    Two modes:
      - If matches and keypoints provided: recover poses from Essential matrix (accurate)
      - Otherwise: build from known servo angles (used only when intrinsics are calibrated)
    """
    K = _camera_intrinsics()
    
    if matches is not None and keypoints_per_frame is not None:
        return _recover_projections(frames, matches, keypoints_per_frame, K)
    
    # Fallback: synthesised from servo angle
    projections = []
    for frame in frames:
        P = _projection_for_pose(frame.pose, K)
        projections.append(P)
    return projections


def _recover_projections(frames, matches, keypoints_per_frame, K) -> list[np.ndarray]:
    """
    Recover camera poses incrementally from matched keypoints.
    Frame 0 is fixed at the world origin facing +Z.
    Each subsequent frame's pose is recovered from its matches against
    the best-connected already-placed frame.
    """
    import cv2
    n = len(frames)
    projections = [None] * n

    # Frame 0: identity — world origin
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    projections[0] = K @ np.hstack([R0, t0])
    print(f"    frame 00: identity (anchor)")

    placed = {0}

    # Build adjacency: for each unplaced frame, find best match to a placed frame
    for _ in range(n - 1):
        best_pair = None
        best_count = 0

        for (i, j), dmatches in matches.items():
            if i in placed and j not in placed:
                candidate = (i, j, False)
            elif j in placed and i not in placed:
                candidate = (j, i, True)  # swapped
            else:
                continue

            if len(dmatches) > best_count:
                best_count = len(dmatches)
                best_pair = candidate

        if best_pair is None:
            break

        placed_idx, new_idx, swapped = best_pair
        key = (min(placed_idx, new_idx), max(placed_idx, new_idx))
        dmatches = matches[key]

        i_key, j_key = min(placed_idx, new_idx), max(placed_idx, new_idx)
        kps_i = keypoints_per_frame[i_key]
        kps_j = keypoints_per_frame[j_key]

        pts_placed = np.float32([kps_i[m.queryIdx].pt for m in dmatches])
        pts_new    = np.float32([kps_j[m.trainIdx].pt for m in dmatches])

        if swapped:
            pts_placed, pts_new = pts_new, pts_placed

        E, mask = cv2.findEssentialMat(pts_placed, pts_new, K,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print(f"    frame {new_idx:02d}: Essential matrix failed — skipping")
            projections[new_idx] = projections[placed_idx]  # fallback
            placed.add(new_idx)
            continue

        _, R, t, _ = cv2.recoverPose(E, pts_placed, pts_new, K, mask=mask)

        # Chain: new pose = placed_pose composed with recovered R, t
        P_placed = projections[placed_idx]
        R_placed = P_placed[:, :3]   # approximation — proper decomp below
        
        # Decompose placed projection to get R_prev, t_prev
        R_prev, t_prev = _decompose_projection(P_placed, K)
        R_new = R @ R_prev
        t_new = R @ t_prev + t
        projections[new_idx] = K @ np.hstack([R_new, t_new])
        placed.add(new_idx)
        print(f"    frame {new_idx:02d}: recovered from frame {placed_idx:02d}  ({len(dmatches)} matches)")

    # Fill any unplaced frames with their neighbour's projection
    for i in range(n):
        if projections[i] is None:
            projections[i] = projections[i - 1]
            print(f"    frame {i:02d}: no matches found — copied from frame {i-1:02d}")

    return projections


def _decompose_projection(P: np.ndarray, K: np.ndarray):
    """Extract R, t from a projection matrix P = K[R|t]."""
    Rt = np.linalg.inv(K) @ P   # 3×4
    R = Rt[:, :3]
    t = Rt[:, 3:4]
    # Enforce valid rotation via SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    return R, t


# --------------------------------------------------------------------------- #
#  Camera intrinsics                                                           #
# --------------------------------------------------------------------------- #

def _camera_intrinsics() -> np.ndarray:
    """
    3x3 intrinsic matrix K for the OV2640 on the ESP32-CAM.

    fx, fy : focal lengths in pixels
    cx, cy : principal point (image centre)

    These are approximate defaults for 800×600 (SVGA).  Replace with values
    from a proper checkerboard calibration for best reconstruction accuracy.

    CALIBRATION NOTE:
        Run OpenCV's calibrateCamera() with a printed checkerboard and a set
        of images from the ESP32-CAM at the target resolution.  Paste the
        resulting fx, fy, cx, cy here and set DIST_COEFFS below.
    """
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT

    # Approximate FOV for OV2640: ~60° horizontal
    fov_h_rad = math.radians(60.0)
    fx = (w / 2.0) / math.tan(fov_h_rad / 2.0)
    fy = fx                        # square pixels assumed

    cx = w / 2.0
    cy = h / 2.0

    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


# Radial/tangential distortion coefficients [k1, k2, p1, p2, k3]
# Zero until a proper calibration is done.
DIST_COEFFS = np.zeros(5, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Per-frame projection                                                        #
# --------------------------------------------------------------------------- #

def _projection_for_pose(pose, K: np.ndarray) -> np.ndarray:
    """
    Builds P = K [R | t] for one CameraPose.

    Camera position in world coordinates:
        If z_position is available:  C = [x, y, z]
        Otherwise:                   C = [x, y, 0]

    The camera points toward the origin, so:
        t = -R @ C
    """
    R = pose.as_rotation_matrix()          # 3×3, derived from Kalman yaw + imu roll/pitch

    xy = pose.xy_position if pose.xy_position is not None \
         else np.array([0.0, 0.0])
    z  = getattr(pose, "z_position", 0.0)
    C  = np.array([xy[0], xy[1], z], dtype=np.float64)

    t = -R @ C                             # translate world origin into camera space

    Rt = np.hstack([R, t.reshape(3, 1)])   # 3×4
    P  = K @ Rt                            # 3×4 projection matrix
    return P