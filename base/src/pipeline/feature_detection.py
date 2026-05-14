"""
Feature Detection — Difference-of-Gaussian keypoints.

Uses OpenCV's SIFT which implements the DoG pyramid described in the wiki:
  - Build Gaussian pyramid
  - Subtract adjacent octave layers -> DoG
  - Find maxima/minima across scale and space -> keypoints

Input:  list of BGR images (np.ndarray, shape H×W×3)
Output: list of lists of cv2.KeyPoint  (one list per frame)
"""

import cv2
import numpy as np


def detect_all(images: list[np.ndarray]) -> list[list[cv2.KeyPoint]]:
    """
    Run DoG keypoint detection on every frame.
    Returns keypoints_per_frame[i] = list of KeyPoint for images[i].
    """
    detector = _build_detector()
    keypoints_per_frame = []

    for i, img in enumerate(images):
        gray = _to_gray(img)
        kps = detector.detect(gray, None)
        keypoints_per_frame.append(kps)
        print(f"    frame {i:02d}: {len(kps)} keypoints")

    return keypoints_per_frame


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #

def _build_detector() -> cv2.SIFT:
    """
    SIFT detector — parameters tuned for small-object orbital scans.

    nfeatures     : soft cap; more keypoints = denser point cloud
    nOctaveLayers : layers per octave in the Gaussian pyramid (default 3)
    contrastThreshold : raise to discard low-contrast keypoints (noise)
    edgeThreshold : raise to keep more edge responses
    sigma         : initial blur applied before pyramid construction
    """
    return cv2.SIFT_create(
        nfeatures=0,           # 0 = unlimited
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)