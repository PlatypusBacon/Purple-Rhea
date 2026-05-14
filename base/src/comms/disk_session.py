import time
from storage.scan_session import ScanSession, ScanFrame, CameraPose
import config

def start_disk_session() -> ScanSession:
    """
    For testing on the bench without the ESP32-CAM.  Loads all JPEGs from disk
    and synthesises poses from config values.
    """
    session = ScanSession()

    for i in range(config.TOTAL_FRAMES):
        angle_deg = i * config.STEP_DEGREES
        filename = f"../input/img_{i:02d}.jpg"
        path = f"{filename}"
        with open(path, "rb") as f:
            image_bytes = f.read()

        pose = CameraPose.bedug_data(servo_angle_deg=angle_deg, radius=config.NOMINAL_RADIUS)
        frame = ScanFrame(index=i, image_bytes=image_bytes, pose=pose)
        session.add_frame(frame)

    return session