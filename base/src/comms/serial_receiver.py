"""
Temporary serial transport — used for bench testing before MQTT is live.
Mirrors the interface that mqtt_receiver.py will eventually expose.
"""

import serial
import time
from storage.scan_session import ScanSession, ScanFrame, CameraPose
from comms.tracker_receiver import open_tracker, TrackerReceiver
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

    tracker = open_tracker()
    if tracker:
        print(f"  tracker: streaming pose from {config.TRACKER_SERIAL_PORT}")

    print("Waiting for board ready...")
    _wait_for_ready(ser)

    frame_index = 0

    while not session.is_complete():
        print(f"\nRequesting frame {frame_index} ({frame_index * config.STEP_DEGREES}°)...")
        ser.write(b"capture\n")

        frame = _receive_one_frame(ser, frame_index, tracker)
        if frame is None:
            print(f"  Frame {frame_index} failed — retrying once")
            frame = _receive_one_frame(ser, frame_index, tracker)

        if frame:
            session.add_frame(frame)
            print(f"  Frame {frame_index} OK — session has {len(session)} frames")
        else:
            print(f"  Frame {frame_index} failed twice — aborting")
            break

        frame_index += 1

    if tracker:
        tracker.stop()
    ser.close()
    return session



def _wait_for_ready(ser: serial.Serial) -> None:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"  board: {line}")
        if line == "STATUS:ready":
            return


def _receive_one_frame(ser: serial.Serial, index: int,
                       tracker: "TrackerReceiver | None" = None) -> ScanFrame | None:
    """
    Reads one IMAGE:<bytes> header then the raw JPEG body.
    Also reads a LOCATION line if the firmware sends one.
    If `tracker` is provided, its latest pose snapshot is used instead
    (the XIAO pose tracker runs on a separate USB CDC-ACM port).
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
            # yaw is read-only on CameraPose (derived from servo_angle_deg or
            # xy_position) — feed it via servo_angle_deg.
            try:
                parts = line.split(":")[1].split(",")
                pose.roll            = float(parts[0])
                pose.pitch           = float(parts[1])
                pose.servo_angle_deg = float(parts[2])
            except (IndexError, ValueError):
                print("  Malformed LOCATION line — using defaults")

        elif line == "STATUS:capture_failed":
            return None

        elif line == "STATUS:frame_done":
            # Snapshot the tracker's latest pose (if any) — this wins over an
            # inline LOCATION line because the XIAO tracker is the source of
            # truth when wired up.
            if tracker is not None:
                sample = tracker.latest()
                if sample is not None:
                    pose.roll            = sample.roll
                    pose.pitch           = sample.pitch
                    pose.servo_angle_deg = sample.yaw  # yaw is a derived property
                else:
                    print("  tracker: no sample yet — falling back to servo angle")

            if image_data:
                return ScanFrame(index=index, image_bytes=bytes(image_data), pose=pose)
            return None