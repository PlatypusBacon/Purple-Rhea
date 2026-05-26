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


# --------------------------------------------------------------------------- #
#  Rig geometry                                                                #
# --------------------------------------------------------------------------- #

def compute_camera_distance(pitch_deg: float) -> float | None:
    """
    Compute 3D slant distance h from camera to turntable centre.

        h = sqrt((r + y·cos θ + sqrt(x² - (H - y·sin θ)²))² + H²)

    pitch_deg — IMU pitch (positive = tilting down toward centre)
    Returns None if inner sqrt argument is negative (physically inconsistent
    rig constants for this angle) — caller should fall back to NOMINAL_RADIUS.
    """
    theta = math.radians(pitch_deg)

    r = config.RIG_BASE_LENGTH
    x = config.RIG_MIDDLE_LENGTH
    y = config.RIG_FINAL_LENGTH
    H = config.CAMERA_HEIGHT

    inner = x**2 - (H - y * math.sin(theta))**2
    if inner < 0:
        return None

    horizontal = r + y * math.cos(theta) + math.sqrt(inner)
    return math.sqrt(horizontal**2 + H**2)


# --------------------------------------------------------------------------- #
#  Public entry point                                                          #
# --------------------------------------------------------------------------- #

def compute_projections(frames, matches=None, keypoints_per_frame=None) -> list[np.ndarray]:
    """
    Two modes:
      - matches + keypoints provided → PnP from frame-0 anchor (accurate)
      - neither provided             → synthesised from servo/IMU angles
    """
    K = _camera_intrinsics()

    if matches is not None and keypoints_per_frame is not None:
        return _recover_projections_pnp(frames, matches, keypoints_per_frame, K)

    projections = []
    for frame in frames:
        pose = frame.pose
        servo_rad = math.radians(pose.servo_angle_deg)
        h   = pose.radius
        H   = config.CAMERA_HEIGHT
        horiz = math.sqrt(max(h**2 - H**2, 0.0))
        print(f"  frame {frame.index:02d}: "
              f"servo={pose.servo_angle_deg:.1f}°  "
              f"imu_yaw={pose.imu_yaw_deg:.1f}°  "
              f"h={h:.4f}m  horiz={horiz:.4f}m  "
              f"C=[{horiz * math.sin(servo_rad):.4f}, "
              f"{horiz * math.cos(servo_rad):.4f}, "
              f"{H:.4f}]")
        P = _projection_for_pose(pose, K)
        projections.append(P)
        look = frame.pose.as_rotation_matrix()[:, 2]
        print(f"  frame {frame.index:02d}: look direction = {look.round(3)}")
    return projections


# --------------------------------------------------------------------------- #
#  PnP pose recovery                                                           #
# --------------------------------------------------------------------------- #

def _recover_projections_pnp(frames, matches, keypoints_per_frame, K):
    """
    Frame 0 = identity anchor.
    Triangulate frame 0 vs each other frame directly,
    then use solvePnPRansac to place every frame independently in the
    same metric coordinate system. No error accumulation across frames.
    """
    import cv2
    n = len(frames)
    projections = [None] * n

    R0, t0 = np.eye(3), np.zeros((3, 1))
    projections[0] = K @ np.hstack([R0, t0])
    print(f"    frame 00: identity (anchor)")

    for i in range(1, n):
        key = (0, i) if (0, i) in matches else None
        if key is None:
            projections[i] = projections[i - 1]
            print(f"    frame {i:02d}: no match to frame 00 — copied from frame {i-1:02d}")
            continue

        dmatches  = matches[key]
        kps_0     = keypoints_per_frame[0]
        kps_i     = keypoints_per_frame[i]

        pts_0 = np.float32([kps_0[m.queryIdx].pt for m in dmatches])
        pts_i = np.float32([kps_i[m.trainIdx].pt for m in dmatches])

        # Triangulate against frame 0 → metric 3D points
        P_init = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        X4d = cv2.triangulatePoints(projections[0], P_init, pts_0.T, pts_i.T)
        X3d = (X4d[:3] / X4d[3]).T  # (M, 3)

        # PnP: place frame i from 3D↔2D correspondences
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X3d, pts_i, K, None,
            iterationsCount=1000, reprojectionError=2.0,
        )
        if not ok or inliers is None or len(inliers) < 6:
            projections[i] = projections[i - 1]
            print(f"    frame {i:02d}: PnP failed ({len(inliers) if inliers is not None else 0} inliers)"
                  f" — copied from frame {i-1:02d}")
            continue

        R, _ = cv2.Rodrigues(rvec)
        projections[i] = K @ np.hstack([R, tvec])
        print(f"    frame {i:02d}: PnP from frame 00  ({len(inliers)} inliers)")

    return projections


# --------------------------------------------------------------------------- #
#  Decompose projection (utility)                                              #
# --------------------------------------------------------------------------- #

def _decompose_projection(P: np.ndarray, K: np.ndarray):
    """Extract R, t from a projection matrix P = K[R|t]."""
    Rt = np.linalg.inv(K) @ P
    R  = Rt[:, :3]
    t  = Rt[:, 3:4]
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
    3×3 intrinsic matrix K for the OV2640 on the ESP32-CAM.
    fx/fy in pixels; principal point at image centre.
    Replace fx with your calibrated value.
    """
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    fx   = 600.0   # ← replace with calibrated value
    fy   = fx      # square pixels assumed
    cx   = w / 2.0
    cy   = h / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


# Radial/tangential distortion coefficients [k1, k2, p1, p2, k3]
# Zero until proper calibration is done.
DIST_COEFFS = np.zeros(5, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Per-frame projection                                                        #
# --------------------------------------------------------------------------- #

def _projection_for_pose(pose, K: np.ndarray) -> np.ndarray:
    servo_rad = math.radians(pose.servo_angle_deg)
    h     = pose.radius   # computed from pitch via compute_camera_distance
    H     = config.CAMERA_HEIGHT
    horiz = math.sqrt(max(h**2 - H**2, 0.0))

    C = np.array([
        horiz * math.sin(servo_rad),
        horiz * math.cos(servo_rad),
        H,
    ], dtype=np.float64)

    # Camera always points inward — no IMU correction needed
    forward  = -C / np.linalg.norm(C)
    world_up = np.array([0., 0., 1.])
    right    = np.cross(forward, world_up)
    right   /= np.linalg.norm(right)
    down     = np.cross(right, forward)
    down    /= np.linalg.norm(down)
    R_world_to_cam = np.column_stack([right, down, forward]).T

    t = -R_world_to_cam @ C

    print(f"    servo={pose.servo_angle_deg:.0f}°  "
          f"pitch={pose.imu_pitch_deg:.1f}°  "
          f"h={h:.4f}m  horiz={horiz:.4f}m  "
          f"C={C.round(3)}")

    return K @ np.hstack([R_world_to_cam, t.reshape(3, 1)])