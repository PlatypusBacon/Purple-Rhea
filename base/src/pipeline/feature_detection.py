"""
Feature Detection — Difference-of-Gaussian keypoints.

Uses OpenCV's SIFT which implements the DoG pyramid described in the wiki:
  - Build Gaussian pyramid
  - Subtract adjacent octave layers -> DoG
  - Find maxima/minima across scale and space -> keypoints

Input:  list of BGR images (np.ndarray, shape HxWx3)
Output: list of lists of cv2.KeyPoint  (one list per frame)
"""

import cv2
import numpy as np


def detect_all(images: list[np.ndarray]) -> list[list[cv2.KeyPoint]]:
    detector = _build_detector()
    keypoints_per_frame = []

    for i, img in enumerate(images):
        gray = _to_gray(img)
        
        # Mask: keep only the central 70% of the image where the object lives.
        # Discards wall/desk background at the edges.
        mask = _centre_mask(gray)
        
        kps = detector.detect(gray, mask)
        keypoints_per_frame.append(kps)
        print(f"    frame {i:02d}: {len(kps)} keypoints")

    return keypoints_per_frame


def _centre_mask(gray: np.ndarray) -> np.ndarray:
    """
    Elliptical mask centred on the image, covering the middle 70%.
    Adjust x_frac/y_frac if the object sits lower/higher in frame.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, int(h * 0.45)   # slightly above centre — object tends to sit here
    axes = (int(w * 0.40), int(h * 0.38))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #

def _build_detector() -> cv2.SIFT:
    import config
    if getattr(config, "LIMIT_MEM", False):
        return cv2.SIFT_create(
            nfeatures=500,         # hard cap per frame — was unlimited
            nOctaveLayers=3,
            contrastThreshold=0.06,  # stricter — fewer weaker keypoints
            edgeThreshold=10,
            sigma=1.6,
        )
    return cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)