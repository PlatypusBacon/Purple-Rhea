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
import os
import config


def detect_all(images: list[np.ndarray]) -> list[list[cv2.KeyPoint]]:
    detector = _build_detector()
    keypoints_per_frame = []

    os.makedirs("output", exist_ok=True)

    for i, img in enumerate(images):
        gray = _to_gray(img)
        mask = _centre_mask(gray)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if config.WHITE:
            white = cv2.inRange(hsv, (0, 0, 210), (180, 50, 255))
        else:
            white = cv2.inRange(hsv, (0, 0, 0), (240, 230, 180))

        # Shrink the white region inward
        kernel = np.ones((8, 8), np.uint8)
        bright_eroded = cv2.erode(white, kernel, iterations=1)

        # Border ring = original white - eroded white = thin strip just inside the white edge
        # These are the dark controller pixels immediately adjacent to the plate
        border_ring = cv2.bitwise_and(white, cv2.bitwise_not(bright_eroded))

        # Remove all white as normal
        mask[white > 0] = 0
        # Then re-enable just the inner edge strip
        mask[border_ring > 0] = 255

        kps = detector.detect(gray, mask)
        keypoints_per_frame.append(kps)
        print(f"    frame {i:02d}: {len(kps)} keypoints")

        # Save the clean binary mask (lossless PNG) for use in triangulation
        cv2.imwrite(f"output/mask_frame{i:02d}.png", mask)

        # Save the debug visualisation (darkened exclusion zones) separately
        debug = img.copy()
        debug[mask == 0] = (0, 0, 80)
        cv2.imwrite(f"output/debug_mask_frame{i:02d}.jpg", debug)

    return keypoints_per_frame


def _centre_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Tight ellipse around just the controller, not the whole plate
    cx, cy = int((w // 2)*1.05), int(h * 0.47)
    axes = (int(w * 0.2), int(h * 0.25))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #

def _build_detector() -> cv2.SIFT:
    import config
    if getattr(config, "LIMIT_MEM", False):
        return cv2.SIFT_create(
            nfeatures=500,
            nOctaveLayers=3,
            contrastThreshold=0.04,
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