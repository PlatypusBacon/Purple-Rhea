from __future__ import annotations

import time
import cv2
import numpy as np

import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose
from pipeline.pose_computation import compute_camera_distance
# reuse the BLE classes from your main session module
from comms.session import BLEPoseClient, _YawFusion, _ImuFilter, _wrap_deg180
import math

def _safe_float(value: float, fallback: float = 0.0, 
                lo: float = -1e6, hi: float = 1e6) -> float:
    """Return value if finite and in range, else fallback."""
    if not math.isfinite(value) or value < lo or value > hi:
        return fallback
    return value

def start_disk_ble_session(on_frame_captured=None) -> ScanSession:
    """
    Debug session: images are loaded from disk (no MQTT/motor),
    but BLE pose requests are still made so the full IMU/yaw fusion
    pipeline can be exercised with real sensor data.
    """
    session        = ScanSession()
    imu_yaw_offset = None
    last_frame_idx = None
    yaw_fusion     = _YawFusion(config.STEP_DEGREES)
    imu_filter     = _ImuFilter(alpha=0.35)

    print("[BLE] connecting...")
    ble = BLEPoseClient()
    print("[BLE] ready")

    try:
        for i in range(config.TOTAL_FRAMES):
            angle_deg = i * config.STEP_DEGREES
            path = f"../../images/cube/capture{(i + 3):02d}.jpg"
            print(f"\n[Frame {i:02d}/{config.TOTAL_FRAMES}]  angle={angle_deg:.1f}°  src={path}")

            # 1. Load image from disk
            with open(path, "rb") as f:
                raw = f.read()
            buf = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2.imdecode failed for {path}")
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            _, enc = cv2.imencode(".jpg", img)
            del img, buf, raw

            # 2. BLE pose — same as live session
            pose_proto = ble.get_latest_pose()

            # Sanity check only — should never be wild with pure accel
            pitch_deg = pose_proto.pitch
            if not math.isfinite(pitch_deg) or not (-90.0 <= pitch_deg <= 90.0):
                print(f"  [WARN] bad pitch {pitch_deg} — using 0.0")
                pitch_deg = 0.0

            print(f"  [BLE] pitch={pitch_deg:.1f}°")
            radius = compute_camera_distance(pitch_deg)

            if last_frame_idx is not None and pose_proto.frame_index == last_frame_idx:
                print(f"  [WARN] pose frame_index did not advance "
                      f"({pose_proto.frame_index}) — XIAO may be stale")
            last_frame_idx = pose_proto.frame_index

            # 3. Yaw fusion (identical logic to live session)
            if imu_yaw_offset is None:
                imu_yaw_offset = pose_proto.yaw
                print(f"  [IMU] yaw offset locked at {imu_yaw_offset:.1f}°")

            fused_yaw_deg, imu_rel_deg, yaw_residual = yaw_fusion.update(
                pose_proto.yaw, angle_deg, i
            )
            print("  [Yaw] cmd={:.1f}°  imu_rel={:.1f}°  residual={:+.1f}°  used={:.1f}°"
                  .format(angle_deg, imu_rel_deg, yaw_residual, fused_yaw_deg))

            filt_pitch_deg, filt_roll_deg = imu_filter.update(
                pose_proto.pitch, pose_proto.roll
            )
            print("  [IMU] pitch raw={:.1f}° filt={:.1f}°  roll raw={:.1f}° filt={:.1f}°"
                  .format(pose_proto.pitch, filt_pitch_deg, pose_proto.roll, filt_roll_deg))

            # 4. Radius (same blending logic)
            h = compute_camera_distance(filt_pitch_deg)
            radius_from_tracker = (
                pose_proto.radius
                if pose_proto.radius_valid and 0.05 <= pose_proto.radius <= 2.0
                else None
            )
            if h is not None and radius_from_tracker is not None:
                radius, radius_source = 0.7 * h + 0.3 * radius_from_tracker, "geometry+imu"
            elif h is not None:
                radius, radius_source = h, "geometry"
            elif radius_from_tracker is not None:
                radius, radius_source = radius_from_tracker, "imu_estimator"
            else:
                radius, radius_source = config.NOMINAL_RADIUS, "nominal"
            print(f"  [Pose] radius={radius:.4f}m  source={radius_source}")

            # 5. Build pose (same as live session, not debug_data)
            pose = CameraPose.from_proto(
                servo_angle_deg = fused_yaw_deg,
                radius          = radius,
                imu_yaw_deg     = pose_proto.yaw,
                imu_pitch_deg   = filt_pitch_deg,
                imu_roll_deg    = filt_roll_deg,
                imu_yaw_offset  = imu_yaw_offset,
            )

            # 6. Store frame
            frame = ScanFrame(index=i, image_bytes=enc.tobytes(), pose=pose)
            session.add_frame(frame)
            del enc

            print(f"  [Session] frame {i} stored ({len(session)}/{config.TOTAL_FRAMES})")
            if on_frame_captured:
                on_frame_captured(i, len(session))

            time.sleep(1.0)   # keep the same BLE pacing as the live session

    finally:
        print("\n[start_disk_ble_session] done — BLE connection will close with BLEPoseClient GC.")

    if not session.is_complete():
        print(f"WARNING: session incomplete — missing frames {session.missing_indices()}")

    print(f"\n[start_disk_ble_session] {len(session)} frames captured.")
    return session