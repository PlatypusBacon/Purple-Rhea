import cv2
import numpy as np
import os
import config
from storage.scan_session import ScanSession, ScanFrame, CameraPose

def start_disk_session() -> ScanSession:
    session = ScanSession()
    image_dir = "../../images/frames"
    
    for i in range(config.TOTAL_FRAMES):
        angle_deg = i * config.STEP_DEGREES
        filename = f"frame_{i:02d}_{int(angle_deg):03d}deg.jpg"
        path = os.path.join(image_dir, filename)
        path = f"/Users/benyin/Desktop/frames/frame_{i:02d}_{int(angle_deg):03d}deg.jpg"

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