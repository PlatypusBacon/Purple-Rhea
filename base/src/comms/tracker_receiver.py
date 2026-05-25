"""
Background reader for the XIAO BLE Sense pose tracker on USB CDC-ACM.

The tracker emits one line per ~50 ms:
    LOCATION:<roll>,<pitch>,<yaw>     (degrees, ZYX)

This module opens the tracker port, spawns a daemon thread that continuously
reads lines and keeps the latest pose timestamped. Consumers call
`latest()` at frame-capture time to attach the most recent pose to the
ScanFrame.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial

import config


@dataclass
class TrackerSample:
    roll: float
    pitch: float
    yaw: float
    timestamp: float       # monotonic seconds when the line was received
    radius: Optional[float] = None  # metres, from RADIUS: lines (may be None)


class TrackerReceiver:
    def __init__(self, port: str, baud: int = config.TRACKER_BAUD_RATE):
        self._port = port
        self._baud = baud
        self._latest: Optional[TrackerSample] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._ser = serial.Serial(self._port, self._baud, timeout=1)
        self._ser.dtr = False
        self._ser.rts = False
        self._thread = threading.Thread(target=self._run, name="tracker-rx",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._ser:
            self._ser.close()

    def latest(self) -> Optional[TrackerSample]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        assert self._ser is not None
        last_radius: Optional[float] = None
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                break

            if line.startswith("RADIUS:"):
                try:
                    last_radius = float(line[len("RADIUS:"):])
                except ValueError:
                    pass
                continue

            if not line.startswith("LOCATION:"):
                continue
            try:
                parts = line[len("LOCATION:"):].split(",")
                roll, pitch, yaw = (float(p) for p in parts[:3])
            except (IndexError, ValueError):
                continue
            sample = TrackerSample(roll=roll, pitch=pitch, yaw=yaw,
                                   timestamp=time.monotonic(),
                                   radius=last_radius)
            with self._lock:
                self._latest = sample


def open_tracker() -> Optional[TrackerReceiver]:
    """Convenience: open the configured tracker port, or return None if
    TRACKER_SERIAL_PORT is unset / the port can't be opened."""
    if not config.TRACKER_SERIAL_PORT:
        return None
    try:
        rx = TrackerReceiver(config.TRACKER_SERIAL_PORT)
        rx.start()
        return rx
    except (serial.SerialException, OSError) as e:
        print(f"  tracker: could not open {config.TRACKER_SERIAL_PORT}: {e}")
        return None
