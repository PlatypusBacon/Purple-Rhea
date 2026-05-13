import serial
import time

ser = serial.Serial("COM15", 115200, timeout=5)

# Disable DTR/RTS to prevent reset on connect
ser.dtr = False
ser.rts = False

print("Waiting for board to be ready...")

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if line:
        print(f"  board: {line}")
    if line == "STATUS:ready":
        print("Sending capture command...")
        ser.write(b"capture\n")
        break

image_data = bytearray()

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if line:
        print(f"  board: {line}")

    if line.startswith("IMAGE:"):
        byte_count = int(line.split(":")[1])
        print(f"Receiving {byte_count} bytes...")

        remaining = byte_count
        while remaining > 0:
            chunk = ser.read(min(remaining, 4096))
            if not chunk:
                print("Timeout!")
                break
            image_data.extend(chunk)
            remaining -= len(chunk)
            print(f"  {len(image_data)}/{byte_count} bytes", end="\r")

        print()
        if len(image_data) == byte_count:
            with open("capture.jpg", "wb") as f:
                f.write(image_data)
            print(f"Saved capture.jpg ({byte_count} bytes)")
        else:
            print(f"Incomplete: {len(image_data)}/{byte_count} bytes")
        break

    elif line == "STATUS:capture_failed":
        print("Camera capture failed on device")
        break