from __future__ import annotations

import time
import cv2
import numpy as np
import math

import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose
from pipeline.pose_computation import compute_camera_distance
from comms.session import BLEPoseClient


def start_disk_ble_session(on_frame_captured=None) -> ScanSession:
    """
    Debug session: images loaded from disk, BLE pitch read from XIAO.
    No motor, no MQTT.
    """
    session        = ScanSession()
    last_frame_idx = None

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

            # 2. Get latest pose from XIAO
            pose_proto = ble.get_latest_pose()

            if last_frame_idx is not None and pose_proto.frame_index == last_frame_idx:
                print(f"  [WARN] pose frame_index did not advance "
                      f"({pose_proto.frame_index}) — XIAO may be stale")
            last_frame_idx = pose_proto.frame_index

            pitch_deg = pose_proto.pitch
            if not math.isfinite(pitch_deg) or not (-90.0 <= pitch_deg <= 90.0):
                print(f"  [WARN] bad pitch {pitch_deg:.1f}° — using 0.0")
                pitch_deg = 0.0
            print(f"  [BLE] pitch={pitch_deg:.1f}°  frame={pose_proto.frame_index}")

            # 3. Compute radius from pitch
            radius = compute_camera_distance(pitch_deg)
            if radius is None:
                radius = config.NOMINAL_RADIUS
                print(f"  [Pose] radius=nominal ({radius:.4f}m)")
            else:
                print(f"  [Pose] radius={radius:.4f}m")

            # 4. Build pose
            pose = CameraPose.from_proto(
                servo_angle_deg = angle_deg,
                radius          = radius,
                imu_yaw_deg     = 0.0,
                imu_pitch_deg   = pitch_deg,
                imu_roll_deg    = 0.0,
                imu_yaw_offset  = 0.0,
            )

            # 5. Store frame
            frame = ScanFrame(index=i, image_bytes=enc.tobytes(), pose=pose)
            session.add_frame(frame)
            del enc

            print(f"  [Session] frame {i} stored ({len(session)}/{config.TOTAL_FRAMES})")
            if on_frame_captured:
                on_frame_captured(i, len(session))

            time.sleep(1.0)

    finally:
        print("\n[start_disk_ble_session] done.")

    if not session.is_complete():
        print(f"WARNING: session incomplete — missing frames {session.missing_indices()}")

    print(f"\n[start_disk_ble_session] {len(session)} frames captured.")
    return session