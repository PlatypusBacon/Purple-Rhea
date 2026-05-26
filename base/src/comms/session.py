from __future__ import annotations

import asyncio
import threading
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import os
import math

from bleak import BleakClient, BleakScanner
import lgpio

import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose
from pipeline.pose_computation import compute_camera_distance
from tracker_pose_pb2 import TrackerPose

# ── BLE config ────────────────────────────────────────────────────────────────
_BLE_DEVICE_NAME   = "TrackerPose"
_BLE_POSE_CHR_UUID = "12345678-1234-5678-1234-56789abcdef1"

# ── Stepper motor ─────────────────────────────────────────────────────────────
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


def _step_motor(chip: int, n_steps: int, phase: int = 0) -> int:
    next_t = time.perf_counter()
    for _ in range(n_steps):
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


# ── BLE pose client ───────────────────────────────────────────────────────────

class BLEPoseClient:
    """
    Subscribes to 20 Hz notify stream from XIAO.
    Call get_latest_pose() to snapshot the most recent frame.
    """

    def __init__(self) -> None:
        self._client: BleakClient | None = None
        self._loop = asyncio.new_event_loop()
        self._latest_proto: TrackerPose | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("BLE: timed out connecting to TrackerPose")

    def _run(self) -> None:
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        while True:
            print("[BLE] scanning for 'TrackerPose'...")
            device = await BleakScanner.find_device_by_name(_BLE_DEVICE_NAME, timeout=15.0)
            if device is None:
                print("[BLE] not found — retrying in 5 s")
                await asyncio.sleep(5)
                continue

            print(f"[BLE] found {device.address} — connecting")
            try:
                async with BleakClient(device) as client:
                    self._client = client
                    await client.start_notify(_BLE_POSE_CHR_UUID, self._on_notify)
                    print("[BLE] connected and subscribed")
                    await asyncio.sleep(0.5)
                    self._ready.set()
                    while client.is_connected:
                        await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[BLE] disconnected: {e} — reconnecting in 3 s")
                self._client = None
                self._ready.clear()
                await asyncio.sleep(3)

    def _on_notify(self, sender, data: bytearray) -> None:
        proto = TrackerPose()
        try:
            proto.ParseFromString(bytes(data))
        except Exception as e:
            print(f"[BLE] decode error: {e}")
            return
        with self._lock:
            self._latest_proto = proto

    def get_latest_pose(self) -> TrackerPose:
        with self._lock:
            proto = self._latest_proto
        if proto is None:
            raise RuntimeError("BLE: no pose received yet")
        return proto


# ── MQTT image waiter ─────────────────────────────────────────────────────────

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

    def wait(self, timeout: float = 30.0) -> bytes:
        if not self._event.wait(timeout):
            raise TimeoutError(f"No image received within {timeout}s")
        self._event.clear()
        data, self._data = self._data, None
        return data


# ── Public entry point ────────────────────────────────────────────────────────

def start_session(on_frame_captured=None) -> ScanSession:
    if os.path.isdir(config.IMAGE_CACHE):
        for fname in os.listdir(config.IMAGE_CACHE):
            fpath = os.path.join(config.IMAGE_CACHE, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        print(f"[start_session] cleared {config.IMAGE_CACHE}")

    session = ScanSession()
    waiter  = _ImageWaiter()

    print("[BLE] connecting...")
    ble = BLEPoseClient()
    print("[BLE] ready")

    def _on_message(client, userdata, msg):
        if msg.topic != _TOPIC_IMAGE:
            return
        data = bytes(msg.payload)
        if len(data) < 4:
            print("  [MQTT] payload too short — discarding")
            return
        pose_len = data[0] | (data[1] << 8)
        jpeg_start = 2 + pose_len
        if jpeg_start >= len(data):
            print("  [MQTT] no JPEG after pose prefix — discarding")
            return
        waiter.set(data[jpeg_start:])

    mqttc = mqtt.Client(client_id="rpi-scanner", protocol=mqtt.MQTTv311)
    mqttc.on_message = _on_message
    mqttc.username_pw_set("47484333", "47484333")
    mqttc.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
    mqttc.subscribe(_TOPIC_IMAGE, qos=0)
    mqttc.loop_start()

    chip = lgpio.gpiochip_open(0)
    for pin in _PINS:
        lgpio.gpio_claim_output(chip, pin, 0)

    steps_per_frame_f = _STEPS_PER_DEGREE * config.STEP_DEGREES
    _motor_phase      = 0
    _accumulator      = 0.0
    last_frame_index  = None

    try:
        for i in range(config.TOTAL_FRAMES):
            angle_deg = i * config.STEP_DEGREES
            print(f"\n[Frame {i:02d}/{config.TOTAL_FRAMES}]  angle={angle_deg:.1f}°")

            # 1. Flush stale image, trigger camera
            waiter.flush()
            mqttc.publish(_TOPIC_TRIGGER, payload=b"1", qos=0)
            print("  [MQTT] trigger sent")

            # 2. Get latest pose from XIAO
            pose_proto = ble.get_latest_pose()

            if last_frame_index is not None and pose_proto.frame_index == last_frame_index:
                print(f"  [WARN] pose frame_index did not advance "
                      f"({pose_proto.frame_index}) — XIAO may be stale")
            last_frame_index = pose_proto.frame_index

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

            # 5. Wait for JPEG
            jpeg_bytes = waiter.wait(timeout=30.0)
            print(f"  [MQTT] image received — {len(jpeg_bytes):,}B")

            # 6. Decode, resize, re-encode
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

            print(f"  [Session] frame {i} stored ({len(session)}/{config.TOTAL_FRAMES})")
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