"""
start_session.py
================
Runs a full 360° scan by:
  1. Triggering the ESP32-CAM via MQTT (topic: SMILE)
  2. Receiving the JPEG on MQTT (topic: PICTURE)
  3. Loading it into a ScanFrame / ScanSession
  4. Stepping the stepper motor 10° via lgpio
  5. Repeating for all 36 frames

Replaces the old debug file-reader version of start_session().
"""

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

# ── Stepper motor ──────────────────────────────────────────────────────────────
IN1, IN2, IN3, IN4 = 17, 18, 27, 22
_PINS = (IN1, IN2, IN3, IN4)

_STEP_DELAY_S = 0.0013  # 1.3 ms per half-step

# Half-step sequence (8 steps = smoothest motion)
_STEP_SEQ = (
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
)

# 28BYJ-48 half-step output
_STEPS_PER_REV = 4096

# External gear reduction on the turntable
_DRIVER_TEETH = 15
_DRIVEN_TEETH = 115
_GEAR_RATIO   = _DRIVEN_TEETH / _DRIVER_TEETH          # 7.6667

# Half-steps needed to rotate the OUTPUT gear one full revolution
_STEPS_PER_OUTPUT_REV = _STEPS_PER_REV * _GEAR_RATIO   # ~31 403

_STEPS_PER_DEGREE = _STEPS_PER_OUTPUT_REV / 360.0      # ~87.2


def _step_motor(chip: int, n_steps: int, phase: int = 0) -> int:
    """Advance the motor n_steps half-steps. Returns the next phase index."""
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
    """De-energise all coils to prevent heating."""
    for pin in _PINS:
        lgpio.gpio_write(chip, pin, 0)


# ── MQTT helpers ───────────────────────────────────────────────────────────────
_TOPIC_TRIGGER  = "SMILE"    # publish → ESP32-CAM takes a photo
_TOPIC_IMAGE    = "PICTURE"  # subscribe → ESP32-CAM sends JPEG bytes


class _ImageWaiter:
    """Thread-safe container: blocks until one JPEG payload arrives."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._data: bytes | None = None

    def set(self, data: bytes) -> None:
        self._data = data
        self._event.set()

    def wait(self, timeout: float = 30.0) -> bytes:
        if not self._event.wait(timeout):
            raise TimeoutError(
                f"No image received from ESP32-CAM within {timeout}s"
            )
        self._event.clear()
        data, self._data = self._data, None
        return data


# ── Public entry point ─────────────────────────────────────────────────────────

def start_session() -> ScanSession:
    """
    Capture TOTAL_FRAMES images at STEP_DEGREES intervals via the ESP32-CAM
    and stepper motor, returning a fully populated ScanSession.
    """
    # Clear existing jpegs from the cache dir without removing the dir itself
    if os.path.isdir(config.IMAGE_CACHE):
        for fname in os.listdir(config.IMAGE_CACHE):
            fpath = os.path.join(config.IMAGE_CACHE, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        print(f"[start_session] cleared {config.IMAGE_CACHE}")

    session = ScanSession()
    waiter  = _ImageWaiter()

    # ── MQTT client setup ──────────────────────────────────────────────────
    def _on_message(client, userdata, msg):
        if msg.topic == _TOPIC_IMAGE:
            print(f"  [MQTT] image received ({len(msg.payload):,} bytes)")
            waiter.set(bytes(msg.payload))

    mqttc = mqtt.Client(client_id="rpi-scanner", protocol=mqtt.MQTTv311)
    mqttc.on_message = _on_message
    mqttc.username_pw_set("47484333", "47484333")

    mqttc.connect(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
    mqttc.subscribe(_TOPIC_IMAGE, qos=0)
    mqttc.loop_start()

    # ── lgpio stepper setup ────────────────────────────────────────────────
    chip = lgpio.gpiochip_open(0)
    for pin in _PINS:
        lgpio.gpio_claim_output(chip, pin, 0)

    steps_per_frame_f = _STEPS_PER_DEGREE * config.STEP_DEGREES  # ~872.6 — keep fractional
    _motor_phase      = 0    # continuous phase across all increments
    _accumulator      = 0.0  # fractional-step accumulator to prevent drift

    try:
        for i in range(config.TOTAL_FRAMES):
            angle_deg = i * config.STEP_DEGREES
            print(f"\n[Frame {i:02d}/{config.TOTAL_FRAMES}]  angle={angle_deg:.1f}°")

            # 1. Trigger camera
            mqttc.publish(_TOPIC_TRIGGER, payload=b"1", qos=0)
            print("  [MQTT] trigger sent")

            # 2. Wait for JPEG
            raw_jpeg = waiter.wait(timeout=30.0)

            # 3. Decode, resize, re-encode
            buf = np.frombuffer(raw_jpeg, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2.imdecode failed on frame {i}")
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            _, enc = cv2.imencode(".jpg", img)
            del img, buf

            # 4. Build pose & frame, add to session, save jpeg to cache
            pose = CameraPose.bedug_data(
                servo_angle_deg=angle_deg,
                radius=config.NOMINAL_RADIUS,
            )
            frame = ScanFrame(index=i, image_bytes=enc.tobytes(), pose=pose)
            session.add_frame(frame)
            frame.save_jpeg(config.IMAGE_CACHE)
            del enc

            print(f"  [Session] frame {i} stored  "
                  f"({len(session)}/{config.TOTAL_FRAMES} total)")

            # 5. Advance motor (skip after last frame)
            if i < config.TOTAL_FRAMES - 1:
                print(f"  [Motor] stepping {config.STEP_DEGREES:.0f}° …")
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
        missing = session.missing_indices()
        print(f"WARNING: session incomplete — missing frames {missing}")

    print(f"\n[start_session] done — {len(session)} frames captured.")
    return session