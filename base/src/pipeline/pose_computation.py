"""
Pose Computation — builds a 3x4 projection matrix P = K [R | t] for each
frame, using the known camera intrinsics and the pose from the Kalman filter
/ servo angle.

FIXES applied vs original:
  - compute_camera_distance() is now actually called and used; if it returns
    None or the tracker radius is zero/invalid, we fall back to NOMINAL_RADIUS
    with a warning rather than silently using a zero/bad radius.
  - PnP branch: triangulation now uses projections[0] (the correct anchor),
    not a freshly constructed identity P_init.
  - Added extensive debug prints throughout so you can see exactly what pose
    values and camera centres each frame produces.
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
        print(f"    [WARN] compute_camera_distance: inner sqrt negative "
              f"(pitch={pitch_deg:.1f}°, inner={inner:.6f}) — "
              f"using NOMINAL_RADIUS={config.NOMINAL_RADIUS:.4f}m")
        return None

    horizontal = r + y * math.cos(theta) + math.sqrt(inner)
    h = math.sqrt(horizontal**2 + H**2)
    print(f"    [pose] pitch={pitch_deg:.2f}° → horiz={horizontal:.4f}m  h={h:.4f}m")
    return h


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
    print(f"\n[pose] Intrinsics K:\n{K}\n")

    if matches is not None and keypoints_per_frame is not None:
        print("[pose] Using PnP mode")
        return _recover_projections_pnp(frames, matches, keypoints_per_frame, K)

    print("[pose] Using servo/IMU synthesis mode")
    projections = []
    for frame in frames:
        pose = frame.pose
        servo_rad = math.radians(pose.servo_angle_deg)

        # ------------------------------------------------------------------ #
        # FIX: actually use compute_camera_distance with the IMU pitch,       #
        #      and fall back to NOMINAL_RADIUS when it fails or radius=0.     #
        # ------------------------------------------------------------------ #
        tracker_radius = pose.radius
        if tracker_radius > 0.01:
            # Tracker provided a plausible radius
            h = tracker_radius
            print(f"  frame {frame.index:02d}: using tracker radius h={h:.4f}m")
        else:
            # Tracker radius zero/invalid — compute from rig geometry + pitch
            h = compute_camera_distance(pose.imu_pitch_deg)
            if h is None:
                h = config.NOMINAL_RADIUS
                print(f"  frame {frame.index:02d}: rig formula failed, "
                      f"using NOMINAL_RADIUS={h:.4f}m")
            else:
                print(f"  frame {frame.index:02d}: computed h={h:.4f}m from pitch={pose.imu_pitch_deg:.1f}°")

        H     = config.CAMERA_HEIGHT
        horiz = math.sqrt(max(h**2 - H**2, 0.0))

        Cx = horiz * math.sin(servo_rad)
        Cy = horiz * math.cos(servo_rad)
        Cz = H

        print(f"  frame {frame.index:02d}: "
              f"servo={pose.servo_angle_deg:.1f}°  "
              f"imu_yaw={pose.imu_yaw_deg:.1f}°  "
              f"imu_pitch={pose.imu_pitch_deg:.1f}°  "
              f"h={h:.4f}m  horiz={horiz:.4f}m  "
              f"C=[{Cx:.4f}, {Cy:.4f}, {Cz:.4f}]")

        # Sanity check: camera should not be at the origin
        C = np.array([Cx, Cy, Cz])
        if np.linalg.norm(C) < 0.01:
            print(f"  [WARN] frame {frame.index:02d}: camera centre is near origin! "
                  f"Check radius/height values. h={h:.4f}, H={H:.4f}, horiz={horiz:.4f}")

        P = _projection_for_pose_with_h(pose, K, h)
        projections.append(P)

        # Decompose and print look direction for debugging
        Rt = np.linalg.inv(K) @ P
        R_check = Rt[:, :3]
        t_check = Rt[:, 3]
        look = R_check[2, :]          # third row = forward direction in cam space
        C_recover = -R_check.T @ t_check
        print(f"  frame {frame.index:02d}: look direction = {look.round(4)}, "
              f"recovered C = {C_recover.round(4)}")

    print(f"\n[pose] All {len(projections)} projections computed.\n")
    return projections


# --------------------------------------------------------------------------- #
#  PnP pose recovery                                                           #
# --------------------------------------------------------------------------- #

def _recover_projections_pnp(frames, matches, keypoints_per_frame, K):
    import cv2
    n = len(frames)
    projections = [None] * n

    R0, t0 = np.eye(3), np.zeros((3, 1))
    projections[0] = K @ np.hstack([R0, t0])
    print(f"    frame 00: identity (anchor)")
    print(f"    anchor P:\n{projections[0]}")

    # ── NEW: servo priors so triangulation has two distinct projections ──
    servo_projections = [
        _projection_for_pose_with_h(
            f.pose, K,
            f.pose.radius if f.pose.radius > 0.01 else config.NOMINAL_RADIUS
        )
        for f in frames
    ]

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
        print(f"    frame {i:02d}: {len(dmatches)} matches to frame 00")

        # ── FIX: use servo prior for frame i as second projection ──
        X4d = cv2.triangulatePoints(projections[0], servo_projections[i], pts_0.T, pts_i.T)
        X3d = (X4d[:3] / X4d[3]).T


        print(f"    frame {i:02d}: triangulated {len(X3d)} 3D points, "
              f"depth range: {X3d[:, 2].min():.4f}..{X3d[:, 2].max():.4f}")

        # Filter points with negative/zero depth (behind camera)
        valid_depth = X3d[:, 2] > 0
        X3d_valid  = X3d[valid_depth]
        pts_i_valid = pts_i[valid_depth]
        print(f"    frame {i:02d}: {valid_depth.sum()} / {len(X3d)} points have positive depth")

        if len(X3d_valid) < 6:
            projections[i] = projections[i - 1]
            print(f"    frame {i:02d}: too few valid 3D points ({len(X3d_valid)}) "
                  f"— copied from frame {i-1:02d}")
            continue

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X3d_valid, pts_i_valid, K, None,
            iterationsCount=1000, reprojectionError=2.0,
        )
        if not ok or inliers is None or len(inliers) < 6:
            projections[i] = projections[i - 1]
            print(f"    frame {i:02d}: PnP failed "
                  f"({len(inliers) if inliers is not None else 0} inliers) "
                  f"— copied from frame {i-1:02d}")
            continue

        R, _ = cv2.Rodrigues(rvec)
        projections[i] = K @ np.hstack([R, tvec])
        C_pnp = -R.T @ tvec.reshape(3)
        print(f"    frame {i:02d}: PnP OK  ({len(inliers)} inliers)  "
              f"camera centre ≈ {C_pnp.round(3)}")

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

    OV2640 at UXGA (1600×1200): typical FOV ≈ 65–78° horizontal.
    For 68° HFOV: fx = (W/2) / tan(HFOV/2) ≈ 800 / tan(34°) ≈ 1185 px
    The original value of 600 gives HFOV ≈ 106° which is too wide for OV2640.
    Replace fx with your calibrated value; 1185 is a better starting estimate.
    """
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    fx   = 1185.0  # ← updated estimate; replace with calibrated value
    fy   = fx      # square pixels assumed
    cx   = w / 2.0
    cy   = h / 2.0
    print(f"[pose] Using fx=fy={fx:.1f}, cx={cx}, cy={cy} "
          f"(image {w}×{h})")
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


# Radial/tangential distortion coefficients [k1, k2, p1, p2, k3]
# Zero until proper calibration is done.
DIST_COEFFS = np.zeros(5, dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Per-frame projection (internal — takes explicit h)                         #
# --------------------------------------------------------------------------- #

def _projection_for_pose_with_h(pose, K: np.ndarray, h: float) -> np.ndarray:
    """
    Build projection matrix P = K [R | t] for one frame.

    Coordinate system:
      - Origin: centre of turntable plate
      - Z axis: up
      - At servo_angle=0 the camera sits on the +Y axis

    h     = slant distance from camera to world origin.
    H     = CAMERA_HEIGHT (physical rig constant).
    theta = atan2(H, horiz) — tilt angle below horizontal, derived from h and H.

    Camera position: C = [horiz·sin a, horiz·cos a, H]
    Optical axis:    radial inward at angle theta below horizontal.
    """
    a = math.radians(pose.servo_angle_deg)
    H     = config.CAMERA_HEIGHT
    horiz = math.sqrt(max(h**2 - H**2, 0.0))
    theta = math.atan2(H, horiz)

    C = np.array([horiz * math.sin(a), horiz * math.cos(a), H], dtype=np.float64)

    cos_a, sin_a = math.cos(a), math.sin(a)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # OpenCV camera axes in world coordinates (X=right, Y=down, Z=forward).
    # Derived by placing the camera on +Y at a=0 then rotating about Z.
    right   = np.array([-cos_a,          sin_a,          0.0   ], dtype=np.float64)
    down    = np.array([ sin_a * sin_t,  cos_a * sin_t, -cos_t ], dtype=np.float64)
    forward = np.array([-sin_a * cos_t, -cos_a * cos_t, -sin_t ], dtype=np.float64)

    R = np.stack([right, down, forward], axis=0)
    t = -R @ C

    print(f"    [proj] servo={pose.servo_angle_deg:.0f}°  "
          f"theta={math.degrees(theta):.1f}°  "
          f"h={h:.4f}m  horiz={horiz:.4f}m  "
          f"C={C.round(3)}  t={t.round(3)}")

    return K @ np.hstack([R, t.reshape(3, 1)])

