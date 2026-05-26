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
        seq = _STEP_SEQ[phase]
        for pin, val in zip(_PINS, seq):
            lgpio.gpio_write(chip, pin, val)
    return phase


def _motor_off(chip: int) -> None:
    for pin in _PINS:
        lgpio.gpio_write(chip, pin, 0)


# ── BLE pose client ───────────────────────────────────────────────────────────

class BLEPoseClient:
    def __init__(self) -> None:
        self._client: BleakClient | None = None
        self._loop = asyncio.new_event_loop()
        self._latest_proto: TrackerPose | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        # FIX 2: separate event that fires on the first real notify packet,
        # so _ready is never set prematurely before data is actually flowing.
        self._pose_event = threading.Event()
        self._stop_event = asyncio.Event()
        # FIX 1: asyncio lock prevents concurrent BleakScanner calls that
        # cause "Operation already in progress" errors on reconnect.
        self._scan_lock: asyncio.Lock | None = None   # created inside the loop
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            self.stop()
            raise RuntimeError("BLE: timed out connecting to TrackerPose")

    def _run(self) -> None:
        # FIX 1: create the lock inside the loop thread so it belongs to the
        # correct event loop (asyncio locks are loop-bound).
        self._scan_lock = asyncio.Lock()
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self) -> None:
        while not self._stop_event.is_set():
            print("[BLE] scanning for 'TrackerPose'...")
            # FIX 1: serialise scanner calls so a reconnect attempt never
            # overlaps with an in-progress scan.
            async with self._scan_lock:
                try:
                    device = await BleakScanner.find_device_by_name(
                        _BLE_DEVICE_NAME, timeout=15.0
                    )
                except Exception as e:
                    print(f"[BLE] scan error: {e} — retrying in 5 s")
                    await asyncio.sleep(5)
                    continue

            if device is None:
                print("[BLE] not found — retrying in 5 s")
                await asyncio.sleep(5)
                continue

            print(f"[BLE] found {device.address} — connecting")
            try:
                async with BleakClient(device) as client:
                    self._client = client
                    await client.start_notify(_BLE_POSE_CHR_UUID, self._on_notify)
                    print("[BLE] connected and subscribed — waiting for first packet")
                    # FIX 2: _ready is now set inside _on_notify when the first
                    # real packet arrives, not here. Remove the premature set.
                    while client.is_connected and not self._stop_event.is_set():
                        await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[BLE] disconnected: {e} — reconnecting in 3 s")
                self._client = None
                self._ready.clear()
                self._pose_event.clear()
                if not self._stop_event.is_set():
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
        # FIX 2: signal that a real packet has arrived; set _ready on the
        # very first packet so callers know data is genuinely flowing.
        self._pose_event.set()
        if not self._ready.is_set():
            print("[BLE] first packet received — marking ready")
            self._ready.set()

    # FIX 3: block until a pose is available instead of raising immediately.
    # This eliminates "no pose received yet" errors when the session starts
    # before the XIAO has sent its first notification.
    def get_latest_pose(self, timeout: float = 5.0) -> TrackerPose:
        if not self._pose_event.wait(timeout=timeout):
            raise TimeoutError(f"BLE: no pose received within {timeout}s")
        with self._lock:
            return self._latest_proto

    def stop(self) -> None:
        """Signal the async loop to exit cleanly."""
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)
        print("[BLE] stopped")


# ── Module-level singleton so start_session() can't double-init ───────────────
_ble_client: BLEPoseClient | None = None
_ble_lock = threading.Lock()

def _get_ble_client() -> BLEPoseClient:
    global _ble_client
    with _ble_lock:
        if _ble_client is None or not _ble_client._thread.is_alive():
            _ble_client = BLEPoseClient()
        return _ble_client


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
    ble = _get_ble_client()
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
    # FIX 5: defensively free each pin before claiming it. lgpio raises
    # "Operation already in progress" if a pin was left claimed from a
    # previous session that exited uncleanly (e.g. mid-frame exception).
    for pin in _PINS:
        try:
            lgpio.gpio_free(chip, pin)
        except Exception:
            pass
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
            # FIX 4: spin-wait up to 2 s for the frame_index to advance so
            # we never record a stale pose from the previous capture cycle.
            if last_frame_index is not None:
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    pose_proto = ble.get_latest_pose()
                    if pose_proto.frame_index != last_frame_index:
                        break
                    time.sleep(0.05)
                else:
                    print(f"  [WARN] pose frame_index still {pose_proto.frame_index} "
                          f"after 2 s — XIAO may be stale, proceeding anyway")
            else:
                pose_proto = ble.get_latest_pose()

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