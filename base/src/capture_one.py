"""ESP32-CAM capture every 5 seconds. Usage: python capture_one.py [prefix]

Each capture is saved to images/<prefix><N>.jpg, where N counts up from 0.
"""

import os
import sys
import time
import serial
import config

PORT = config.SERIAL_PORT   # <-- edit me
BAUD = config.BAUD_RATE
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "capture"
OUT_DIR = "images"
INTERVAL_S = 5
counter = 0

os.makedirs(OUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=10)
ser.dtr = False
ser.rts = False

print("Waiting for STATUS:ready...")
while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if line:
        print(f"  board: {line}")
    if line == "STATUS:ready":
        break


def capture_once():
    global counter
    # Drain any buffered STATUS:ready announcements before triggering.
    ser.reset_input_buffer()
    ser.write(b"capture\n")
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print(f"  board: {line}")
        if line.startswith("IMAGE:"):
            n = int(line.split(":")[1])
            buf = bytearray()
            while len(buf) < n:
                buf.extend(ser.read(n - len(buf)))
            out_path = os.path.join(OUT_DIR, f"{PREFIX}{counter}.jpg")
            with open(out_path, "wb") as f:
                f.write(buf)
            print(f"Saved {len(buf)} bytes -> {out_path}")
            counter += 1
            return True
        if line == "STATUS:capture_failed":
            print("Capture failed.")
            return False


try:
    while True:
        start = time.monotonic()
        capture_once()
        elapsed = time.monotonic() - start
        sleep_for = max(0.0, INTERVAL_S - elapsed)
        time.sleep(sleep_for)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    ser.close()
