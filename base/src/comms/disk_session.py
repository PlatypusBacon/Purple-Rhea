import cv2
import numpy as np
import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose

def start_disk_session() -> ScanSession:
    session = ScanSession()
    for i in range(config.TOTAL_FRAMES):
        angle_deg = i * config.STEP_DEGREES
        path = f"../../images/test5/capture{(i+1):02d}.jpg"

        buf = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        _, enc = cv2.imencode(".jpg", img)

        pose = CameraPose.bedug_data(
            servo_angle_deg=angle_deg,
            radius=config.NOMINAL_RADIUS,
        )
        session.add_frame(ScanFrame(index=i, image_bytes=enc.tobytes(), pose=pose))
        del img, enc, buf

    return session