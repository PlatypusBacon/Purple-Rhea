"""
Temporary serial transport — used for bench testing before MQTT is live.
Mirrors the interface that mqtt_receiver.py will eventually expose.
"""

import serial
import time
from storage.scan_session import ScanSession, ScanFrame, CameraPose
import config


def receive_session_serial(port: str = config.SERIAL_PORT,
                           baud: int = config.BAUD_RATE) -> ScanSession:
    """
    Collects all frames as ScanSession. Serial rn for debugging
    """
    session = ScanSession()
    ser = serial.Serial(port, baud, timeout=5)
    ser.dtr = False
    ser.rts = False

    print("Waiting for board ready...")
    _wait_for_ready(ser)

    frame_index = 0

    while not session.is_complete():
        print(f"\nRequesting frame {frame_index} ({frame_index * config.STEP_DEGREES}°)...")
        ser.write(b"capture\n")

        frame = _receive_one_frame(ser, frame_index)
        if frame is None:
            print(f"  Frame {frame_index} failed — retrying once")
            frame = _receive_one_frame(ser, frame_index)

        if frame:
            session.add_frame(frame)
            print(f"  Frame {frame_index} OK — session has {len(session)} frames")
        else:
            print(f"  Frame {frame_index} failed twice — aborting")
            break

        frame_index += 1

    ser.close()
    return session



def _wait_for_ready(ser: serial.Serial) -> None:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  board: {line}")
        if line == "STATUS:ready":
            return


def _receive_one_frame(ser: serial.Serial, index: int) -> ScanFrame | None:
    """
    Reads one IMAGE:<bytes> header then the raw JPEG body.
    Also reads a LOCATION line if the firmware sends one.
    Returns a ScanFrame or None on failure.
    """
    image_data = bytearray()
    pose = CameraPose(servo_angle_deg=index * config.STEP_DEGREES)

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print(f"  board: {line}")

        if line.startswith("IMAGE:"):
            byte_count = int(line.split(":")[1])
            remaining = byte_count
            while remaining > 0:
                chunk = ser.read(min(remaining, 4096))
                if not chunk:
                    print("  Timeout reading image bytes")
                    return None
                image_data.extend(chunk)
                remaining -= len(chunk)
                print(f"  {len(image_data)}/{byte_count} bytes", end="\r")
            print()

            if len(image_data) != byte_count:
                return None

        elif line.startswith("LOCATION:"):
            # Expected format: LOCATION:<roll>,<pitch>,<yaw>
            # Extend this when the tracking node sends richer data
            try:
                parts = line.split(":")[1].split(",")
                pose.roll  = float(parts[0])
                pose.pitch = float(parts[1])
                pose.yaw   = float(parts[2])
            except (IndexError, ValueError):
                print("  Malformed LOCATION line — using defaults")

        elif line == "STATUS:capture_failed":
            return None

        elif line == "STATUS:frame_done":
            # Both IMAGE and LOCATION received for this frame
            if image_data:
                return ScanFrame(index=index, image_bytes=bytes(image_data), pose=pose)
            return None