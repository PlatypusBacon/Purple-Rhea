from __future__ import annotations
from dataclasses import dataclass, field
import config
from typing import Optional
import numpy as np
import math


@dataclass
class CameraPose:
    """
    Pose derived from the XIAO-nRF52840 Kalman filter output.
    Calculates yaw and pitch from the XY position and radius.
    Angles stored in degrees.
    """

    xy_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    ) # Kalman input

    radius: float = config.NOMINAL_RADIUS # Expected radius for correction

    servo_angle_deg: float = config.STEP_DEGREES # Angle the thing moves
    
    z_position: float = config.CAMERA_HEIGHT # Height of the camera above the plane
    roll: float = 0.0 # assuming we arent jiggling
    pitch: float = 0.0 # assuming we arent jiggling
    @classmethod
    def bedug_data(cls, servo_angle_deg: float, radius: float) -> "CameraPose":
        angle_rad = math.radians(servo_angle_deg)
        x = radius * math.sin(angle_rad)
        y = radius * math.cos(angle_rad)
        return cls(
            xy_position=np.array([x, y], dtype=float),
            servo_angle_deg=servo_angle_deg,
        )
    @property
    def yaw(self) -> float:
        """Azimuth derived from Kalman xy. Falls back to servo angle."""
        norm = float(np.linalg.norm(self.xy_position))
        if norm > 1e-6:
            return math.degrees(math.atan2(self.xy_position[0],
                                        self.xy_position[1])) % 360.0
        return self.servo_angle_deg % 360.0

    def as_rotation_matrix(self) -> np.ndarray:
        import math
        import numpy as np
        
        angle_rad = math.radians(self.yaw)
        
        # Camera position
        r = config.NOMINAL_RADIUS
        z = self.z_position
        
        # Direction from camera to origin (the object)
        cam_pos = np.array([r * math.sin(angle_rad), 
                            r * math.cos(angle_rad), 
                            z])
        forward = -cam_pos  # points toward origin
        forward /= np.linalg.norm(forward)
        
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.column_stack([right, -up, forward])
        return R

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
