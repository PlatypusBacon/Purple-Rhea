from __future__ import annotations
from dataclasses import dataclass, field
import config
from typing import Optional
import numpy as np
import math


def _wrap_deg180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


imu_roll_offset:  float = 0.0
imu_pitch_offset: float = 0.0
@dataclass
class CameraPose:
    """
    Pose derived from the XIAO-nRF52840 Kalman filter output.

    servo_angle_deg  — effective turntable yaw used for camera centre position C
    imu_yaw_deg      — tracker body yaw; corrects camera orientation in-mount
    imu_pitch_deg    — tracker body pitch
    imu_roll_deg     — tracker body roll
    xy_position      — Kalman XY (falls back to servo angle if zero)
    """

    xy_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )

    radius: float = config.NOMINAL_RADIUS
    servo_angle_deg: float = config.STEP_DEGREES
    z_position: float = config.CAMERA_HEIGHT

    # IMU body orientation from tracker — separate from turntable angle
    imu_yaw_deg:   float = 0.0
    imu_pitch_deg: float = 0.0
    imu_roll_deg:  float = 0.0

    # imu_yaw_offset: subtracted from imu_yaw_deg to give relative correction
    # Set once at frame 0 in start_session.py
    imu_yaw_offset: float = 0.0

    @classmethod
    def from_proto(
        cls,
        servo_angle_deg: float,
        radius: float,
        imu_yaw_deg: float,
        imu_pitch_deg: float,
        imu_roll_deg: float,
        imu_yaw_offset: float = 0.0,
    ) -> "CameraPose":
        """
        Primary constructor for live capture.
        servo_angle_deg  — effective yaw (commanded angle optionally corrected by IMU)
        imu_*            — raw tracker proto values
        imu_yaw_offset   — yaw at frame 0, subtracted to give relative correction
        """
        angle_rad = math.radians(servo_angle_deg)
        x = radius * math.sin(angle_rad)
        y = radius * math.cos(angle_rad)
        return cls(
            xy_position=np.array([x, y], dtype=float),
            servo_angle_deg=servo_angle_deg,
            radius=radius,
            imu_yaw_deg=imu_yaw_deg,
            imu_pitch_deg=imu_pitch_deg,
            imu_roll_deg=imu_roll_deg,
            imu_yaw_offset=imu_yaw_offset,
        )

    @classmethod
    def bedug_data(cls, servo_angle_deg: float, radius: float) -> "CameraPose":
        """Servo-angle-only path — no tracker data available."""
        angle_rad = math.radians(servo_angle_deg)
        x = radius * math.sin(angle_rad)
        y = radius * math.cos(angle_rad)
        return cls(
            xy_position=np.array([x, y], dtype=float),
            servo_angle_deg=servo_angle_deg,
        )

    # ── Derived angles ────────────────────────────────────────────────────────

    @property
    def yaw(self) -> float:
        """
        Turntable azimuth — used for camera centre position in world space.
        Derived from Kalman xy_position; falls back to servo_angle_deg.
        NOT the IMU yaw — see imu_yaw_corrected for that.
        """
        norm = float(np.linalg.norm(self.xy_position))
        if norm > 1e-6:
            return math.degrees(
                math.atan2(self.xy_position[0], self.xy_position[1])
            ) % 360.0
        return self.servo_angle_deg % 360.0

    @property
    def imu_yaw_corrected(self) -> float:
        """IMU yaw relative to frame-0 reference — camera body correction."""
        return self.imu_yaw_deg - self.imu_yaw_offset

    # ── Rotation matrix ───────────────────────────────────────────────────────

    def as_rotation_matrix(self) -> np.ndarray:
        """
        World-to-camera rotation matrix matching _projection_for_pose_with_h.
        """
        a = math.radians(self.servo_angle_deg)
        h = self.radius

        if abs(self.imu_pitch_deg) > 1.0:
            theta = math.radians(self.imu_pitch_deg)
        else:
            H = config.CAMERA_HEIGHT
            horiz = math.sqrt(max(h**2 - H**2, 0.0))
            theta = math.atan2(H, horiz)

        cos_a, sin_a = math.cos(a), math.sin(a)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        right   = np.array([-cos_a,          sin_a,          0.0   ])
        down    = np.array([ sin_a * sin_t,  cos_a * sin_t, -cos_t ])
        forward = np.array([-sin_a * cos_t, -cos_a * cos_t, -sin_t ])

        return np.stack([right, down, forward], axis=0)

@dataclass
class ScanFrame:
    """
    One captured frame: JPEG bytes + the pose at time of capture.
    index: 0-35, corresponding to 0°-350° in 10° steps.
    """
    index:       int
    image_bytes: bytes
    pose:        CameraPose
    # Populated lazily when the pipeline loads the image
    image_array: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def angle_deg(self) -> float:
        return self.pose.servo_angle_deg

    def load_image(self) -> np.ndarray:
        """Decode JPEG bytes → BGR numpy array (OpenCV format)."""
        import cv2
        buf = np.frombuffer(self.image_bytes, dtype=np.uint8)
        self.image_array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return self.image_array

    def save_jpeg(self, directory: str) -> str:
        """Write the raw JPEG to disk for debugging."""
        import os
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"frame_{self.index:02d}_{int(self.angle_deg):03d}deg.jpg")
        with open(path, "wb") as f:
            f.write(self.image_bytes)
        return path
    
@dataclass
class ScanSession:
    """
    Holds all frames for one complete 360° scan.
    Acts as the handoff object between the comms layer and the pipeline.
    """
    frames: list[ScanFrame] = field(default_factory=list)


    def add_frame(self, frame: ScanFrame) -> None:
        self.frames.append(frame)
        self.frames.sort(key=lambda f: f.index)

    def is_complete(self) -> bool:
        from config import TOTAL_FRAMES
        return len(self.frames) == TOTAL_FRAMES

    def missing_indices(self) -> list[int]:
        from config import TOTAL_FRAMES
        received = {f.index for f in self.frames}
        return [i for i in range(TOTAL_FRAMES) if i not in received]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def save_all_jpegs(self, directory: str = "output/frames") -> None:
        for frame in self.frames:
            path = frame.save_jpeg(directory)
            print(f"  saved {path}")
