from __future__ import annotations

import threading
import time
import cv2
import numpy as np
import lgpio
import paho.mqtt.client as mqtt
import os

import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose
from pipeline.pose_computation import compute_camera_distance
from tracker_pose_pb2 import TrackerPose   # generated from .proto
from google.protobuf.message import DecodeError

# ── Stepper motor ──────────────────────────────────────────────────────────────
IN1, IN2, IN3, IN4 = 17, 18, 27, 22
_PINS = (IN1, IN2, IN3, IN4)
_STEP_DELAY_S = 0.0013

_STEP_SEQ = (
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
    (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1),
)

_STEPS_PER_REV        = 4096
_DRIVER_TEETH         = 15
_DRIVEN_TEETH         = 115
_GEAR_RATIO           = _DRIVEN_TEETH / _DRIVER_TEETH
_STEPS_PER_OUTPUT_REV = _STEPS_PER_REV * _GEAR_RATIO
_STEPS_PER_DEGREE     = _STEPS_PER_OUTPUT_REV / 360.0


def _wrap_deg180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


class _YawFusion:
    """
    Tracks IMU yaw as a continuous relative angle and maps it onto the
    commanded turntable progression.
    """
    def __init__(self, step_deg: float) -> None:
        self._step_deg = step_deg
        self._prev_raw_yaw: float | None = None
        self._rel_unwrapped = 0.0
        self._sign = 1.0
        self._sign_locked = False

    def update(self, raw_yaw_deg: float, expected_deg: float, frame_idx: int):
        if self._prev_raw_yaw is None:
            self._prev_raw_yaw = raw_yaw_deg
            self._rel_unwrapped = 0.0
            return expected_deg, 0.0, 0.0

        step_delta = _wrap_deg180(raw_yaw_deg - self._prev_raw_yaw)
        self._prev_raw_yaw = raw_yaw_deg

        # Reject occasional wrap/glitch jumps that are too large for one frame.
        if abs(step_delta) <= 45.0:
            self._rel_unwrapped += step_delta

        rel = self._rel_unwrapped
        if (not self._sign_locked and frame_idx >= 1 and
                abs(rel) >= (0.5 * self._step_deg)):
            err_pos = abs(_wrap_deg180(rel - expected_deg))
            err_neg = abs(_wrap_deg180(-rel - expected_deg))
            self._sign = -1.0 if err_neg + 1.0 < err_pos else 1.0
            self._sign_locked = True

        imu_rel = self._sign * rel
        residual = _wrap_deg180(imu_rel - expected_deg)
        residual = max(-15.0, min(15.0, residual))
        fused = (expected_deg + residual) % 360.0
        return fused, imu_rel, residual


class _ImuFilter:
    """Simple low-pass filter for IMU pitch/roll used by reconstruction."""
    def __init__(self, alpha: float = 0.35) -> None:
        self._alpha = alpha
        self._pitch: float | None = None
        self._roll: float | None = None

    def update(self, pitch_deg: float, roll_deg: float):
        # Keep values in a physically plausible range for this rig.
        pitch_deg = max(-89.0, min(89.0, pitch_deg))
        roll_deg = max(-89.0, min(89.0, roll_deg))

        if self._pitch is None:
            self._pitch = pitch_deg
            self._roll = roll_deg
        else:
            self._pitch = self._alpha * pitch_deg + (1.0 - self._alpha) * self._pitch
            self._roll = self._alpha * roll_deg + (1.0 - self._alpha) * self._roll

        return self._pitch, self._roll


def _step_motor(chip: int, n_steps: int, phase: int = 0) -> int:
    next_t = time.perf_counter()
    for _ in range(n_steps):
        pattern = _STEP_SEQ[phase]
        for pin, value in zip(_PINS, pattern):
            lgpio.gpio_write(chip, pin, value)
        next_t += _STEP_DELAY_S
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.perf_counter()
        phase = (phase + 1) % 8
    return phase


def _motor_off(chip: int) -> None:
    for pin in _PINS:
        lgpio.gpio_write(chip, pin, 0)


# ── MQTT ───────────────────────────────────────────────────────────────────────
_TOPIC_TRIGGER = "SMILE"
_TOPIC_IMAGE   = "PICTURE"


class _ImageWaiter:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._data  = None

    def flush(self) -> None:
        self._event.clear()
        self._data = None

    def set(self, data) -> None:
        self._data = data
        self._event.set()

    def wait(self, timeout: float = 30.0):
        if not self._event.wait(timeout):
            raise TimeoutError(f"No image received within {timeout}s")
        self._event.clear()
        data, self._data = self._data, None
        return data


# ── Public entry point ─────────────────────────────────────────────────────────

def start_session(on_frame_captured=None) -> ScanSession:
    if os.path.isdir(config.IMAGE_CACHE):
        for fname in os.listdir(config.IMAGE_CACHE):
            fpath = os.path.join(config.IMAGE_CACHE, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        print(f"[start_session] cleared {config.IMAGE_CACHE}")

    session = ScanSession()
    waiter  = _ImageWaiter()

    # ── MQTT ──────────────────────────────────────────────────────────────
    def _on_message(client, userdata, msg):
        if msg.topic != _TOPIC_IMAGE:
            return
        data = bytes(msg.payload)

        # Envelope: [pose_len: 2B LE][proto bytes][jpeg bytes]
        if len(data) < 2:
            print("  [MQTT] payload too short — discarding")
            return
        pose_len   = int.from_bytes(data[0:2], 'little')
        if len(data) < 2 + pose_len:
            print(f"  [MQTT] truncated envelope — discarding "
                  f"(total={len(data)}B, pose_len={pose_len}, "
                  f"need={2 + pose_len}B)")
            return
        pose_bytes = data[2 : 2 + pose_len]
        jpeg_bytes = data[2 + pose_len :]

        pose_proto = TrackerPose()
        try:
            pose_proto.ParseFromString(pose_bytes)
        except DecodeError:
            print("  [MQTT] pose protobuf decode failed — discarding frame")
            return

        print(f"  [MQTT] received — frame={pose_proto.frame_index}  yaw={pose_proto.yaw:.1f}°  "
              f"pitch={pose_proto.pitch:.1f}°  "
              f"jpeg={len(jpeg_bytes):,}B")
        waiter.set((jpeg_bytes, pose_proto))

    mqttc = mqtt.Client(client_id="rpi-scanner", protocol=mqtt.MQTTv311)
    mqttc.on_message = _on_message
    mqttc.username_pw_set("47484333", "47484333")
    mqttc.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
    mqttc.subscribe(_TOPIC_IMAGE, qos=0)
    mqttc.loop_start()

    # ── Stepper ───────────────────────────────────────────────────────────
    chip = lgpio.gpiochip_open(0)
    for pin in _PINS:
        lgpio.gpio_claim_output(chip, pin, 0)

    steps_per_frame_f = _STEPS_PER_DEGREE * config.STEP_DEGREES
    _motor_phase  = 0
    _accumulator  = 0.0
    imu_yaw_offset = None   # set on frame 0
    last_pose_frame_index = None
    yaw_fusion = _YawFusion(config.STEP_DEGREES)
    imu_filter = _ImuFilter(alpha=0.35)

    try:
        for i in range(config.TOTAL_FRAMES):
            angle_deg = i * config.STEP_DEGREES
            print(f"\n[Frame {i:02d}/{config.TOTAL_FRAMES}]  angle={angle_deg:.1f}°")

            # 1. Flush stale data, trigger camera
            waiter.flush()
            mqttc.publish(_TOPIC_TRIGGER, payload=b"1", qos=0)
            print("  [MQTT] trigger sent")

            # 2. Wait for envelope (jpeg + proto)
            jpeg_bytes, pose_proto = waiter.wait(timeout=30.0)

            # 3. Set yaw reference on first frame
            if imu_yaw_offset is None:
                imu_yaw_offset = pose_proto.yaw
                print(f"  [IMU] yaw offset locked at {imu_yaw_offset:.1f}°")

            if last_pose_frame_index is not None and pose_proto.frame_index == last_pose_frame_index:
                print("  [WARN] tracker pose frame_index did not advance "
                      f"({pose_proto.frame_index}) — UART data may be stale")
            last_pose_frame_index = pose_proto.frame_index

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

            # 4. Compute camera-to-centre distance from pitch + rig geometry
            h = compute_camera_distance(filt_pitch_deg)
            radius_from_tracker = (
                pose_proto.radius
                if pose_proto.radius_valid and 0.05 <= pose_proto.radius <= 2.0
                else None
            )
            if h is not None and radius_from_tracker is not None:
                radius = 0.7 * h + 0.3 * radius_from_tracker
                radius_source = "geometry+imu"
            elif h is not None:
                radius = h
                radius_source = "geometry"
            elif radius_from_tracker is not None:
                radius = radius_from_tracker
                radius_source = "imu_estimator"
            else:
                radius = config.NOMINAL_RADIUS
                radius_source = "nominal"
            print(f"  [Pose] radius={radius:.4f}m  source={radius_source}")

            # 5. Build pose
            pose = CameraPose.from_proto(
                servo_angle_deg = fused_yaw_deg,
                radius          = radius,
                imu_yaw_deg     = pose_proto.yaw,
                imu_pitch_deg   = filt_pitch_deg,
                imu_roll_deg    = filt_roll_deg,
                imu_yaw_offset  = imu_yaw_offset,
            )

            # 6. Decode, resize, re-encode JPEG
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2.imdecode failed on frame {i}")
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            _, enc = cv2.imencode(".jpg", img)
            del img, buf

            # 7. Store frame
            frame = ScanFrame(index=i, image_bytes=enc.tobytes(), pose=pose)
            session.add_frame(frame)
            frame.save_jpeg(config.IMAGE_CACHE)
            del enc

            print(f"  [Session] frame {i} stored  "
                  f"({len(session)}/{config.TOTAL_FRAMES} total)")
            if on_frame_captured:
                on_frame_captured(i, len(session))

            # 8. Advance motor (skip after last frame)
            if i < config.TOTAL_FRAMES - 1:
                print(f"  [Motor] stepping {config.STEP_DEGREES:.0f}°")
                _accumulator += steps_per_frame_f
                n = int(_accumulator)
                _accumulator -= n
                _motor_phase = _step_motor(chip, n, _motor_phase)
                _motor_off(chip)

            time.sleep(1.0)

    finally:
        _motor_off(chip)
        for pin in _PINS:
            lgpio.gpio_free(chip, pin)
        lgpio.gpiochip_close(chip)
        mqttc.loop_stop()
        mqttc.disconnect()
        print("\n[start_session] MQTT disconnected, GPIO released.")

    if not session.is_complete():
        print(f"WARNING: session incomplete — missing frames {session.missing_indices()}")

    print(f"\n[start_session] done — {len(session)} frames captured.")
    return session
