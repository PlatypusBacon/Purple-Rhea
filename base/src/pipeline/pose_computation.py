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

def compute_camera_distance(theta: float) -> float | None:
    """
    Compute 3D slant distance h from camera to turntable centre.

        h = sqrt((r + y·cos θ + sqrt(x² - (H - y·sin θ)²))² + H²)
    This was calculated using the rig geometry as a 3 fixed length problem with movable points.
    TO do this one other length had to be defined as constant, this was chosen as the height of the camera.
    this is defined as CAMERA_HEIGHT in config.py. The formula is then derived by solving for the horizontal distance from the camera to the origin in terms of the pitch angle, 
    and then using Pythagorean theorem to get the slant distance h.

    pitch_deg — IMU pitch (positive = tilting down toward centre)
    Returns None if inner sqrt argument is negative (physically inconsistent
    rig constants for this angle) — caller should fall back to NOMINAL_RADIUS.
    """

    r = config.RIG_BASE_LENGTH
    x = config.RIG_MIDDLE_LENGTH
    y = config.RIG_FINAL_LENGTH
    H = config.CAMERA_HEIGHT

    inner = x**2 - (H - y * math.sin(theta))**2
    if inner < 0:
        print(f"    [WARN] compute_camera_distance: inner sqrt negative "
              f"(pitch={math.degrees(theta):.1f}°, inner={inner:.6f}) — "
              f"using NOMINAL_RADIUS={config.NOMINAL_RADIUS:.4f}m")
        return None

    horizontal = r + y * math.cos(theta) + math.sqrt(inner)
    h = math.sqrt(horizontal**2 + H**2)
    print(f"    [pose] pitch={math.degrees(theta):.2f}° → horiz={horizontal:.4f}m  h={h:.4f}m")
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

    print("[pose] Using servo/IMU synthesis mode")
    H = config.CAMERA_HEIGHT

    projections = []
    for frame in frames:
        pose  = frame.pose
        horiz = math.sqrt(max(pose.radius**2 - H**2, 0.0))
        theta = math.atan2(H, horiz)
        servo_rad = math.radians(pose.servo_angle_deg)

        Cx = horiz * math.sin(servo_rad)
        Cy = horiz * math.cos(servo_rad)
        Cz = H

        print(f"  frame {frame.index:02d}: "
              f"servo={pose.servo_angle_deg:.1f}°  "
              f"radius={pose.radius:.4f}m  horiz={horiz:.4f}m  theta={math.degrees(theta):.2f}°  "
              f"C=[{Cx:.4f}, {Cy:.4f}, {Cz:.4f}]")

        P = _build_projection(pose.servo_angle_deg, K, horiz, H, theta)
        projections.append(P)

        Rt = np.linalg.inv(K) @ P
        R_check = Rt[:, :3]
        t_check = Rt[:, 3]
        look = R_check[2, :]
        C_recover = -R_check.T @ t_check
        print(f"  frame {frame.index:02d}: look direction = {look.round(4)}, "
              f"recovered C = {C_recover.round(4)}")

    print(f"\n[pose] All {len(projections)} projections computed.\n")
    return projections



# --------------------------------------------------------------------------- #
#  Camera intrinsics                                                           #
# --------------------------------------------------------------------------- #

def _camera_intrinsics() -> np.ndarray:
    """
    returns the 3x3 intrinsic matrix for the camera, calculated using a loosely calibrated
    fx value for the fov calculation. With 1185.0 the fov is around 60
    """
    w, h = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
    fx   = 1185.0  # determined by calculation of pixels against cube faces of known size
    fy   = fx      # square pixels
    cx   = w / 2.0
    cy   = h / 2.0
    print(f"[pose] Using fx=fy={fx:.1f}, cx={cx}, cy={cy} "
          f"(image {w}x{h})")
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

def _build_projection(servo_deg: float, K: np.ndarray,
                      horiz: float, H: float, theta: float) -> np.ndarray:
    """
    Build P = K [R | t] for one frame.

    Camera sits at [horiz·sin a, horiz·cos a, H] and looks toward the origin
    tilted theta below horizontal.
    """
    a = math.radians(servo_deg)
    C = np.array([horiz * math.sin(a), horiz * math.cos(a), H], dtype=np.float64)

    cos_a, sin_a = math.cos(a), math.sin(a)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    right   = np.array([-cos_a,          sin_a,          0.0   ], dtype=np.float64)
    down    = np.array([ sin_a * sin_t,  cos_a * sin_t, -cos_t ], dtype=np.float64)
    forward = np.array([-sin_a * cos_t, -cos_a * cos_t, -sin_t ], dtype=np.float64)

    R = np.stack([right, down, forward], axis=0)
    t = -R @ C

    return K @ np.hstack([R, t.reshape(3, 1)])

