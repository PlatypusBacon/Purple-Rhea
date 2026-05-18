"""One-shot ESP32-CAM capture. Usage: python capture_one.py [out.jpg]"""

import sys
import serial
import config

PORT = config.SERIAL_PORT   # <-- edit me
BAUD = config.BAUD_RATE
OUT  = sys.argv[1] if len(sys.argv) > 1 else "capture.jpg"

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
        with open(OUT, "wb") as f:
            f.write(buf)
        print(f"Saved {len(buf)} bytes -> {OUT}")
        break
    if line == "STATUS:capture_failed":
        print("Capture failed.")
        break

ser.close()
