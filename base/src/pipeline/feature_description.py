"""
Feature Description — 128-D SIFT descriptor vectors.

For each keypoint, builds a descriptor from the histogram of gradient
orientations in the surrounding neighbourhood (4×4 spatial bins × 8
orientation bins = 128 floats), as described in the wiki.

Input:
    images            : list of BGR np.ndarray
    keypoints_per_frame: list[list[cv2.KeyPoint]]  from feature_detection

Output:
    descriptors_per_frame[i] -> np.ndarray shape (N_i, 128), dtype float32
                                 None if no keypoints found for that frame
"""

import cv2
import numpy as np


def describe_all(
    images: list[np.ndarray],
    keypoints_per_frame: list[list[cv2.KeyPoint]],
) -> list[np.ndarray | None]:
    """
    Compute SIFT descriptors for every frame's keypoints.
    Descriptor computation is done with the same SIFT instance that was used
    for detection so the orientation assignment is consistent.
    """
    describer = _build_describer()
    descriptors_per_frame = []

    for i, (img, kps) in enumerate(zip(images, keypoints_per_frame)):
        if not kps:
            print(f"    frame {i:02d}: no keypoints — skipping descriptor")
            descriptors_per_frame.append(None)
            continue

        gray = _to_gray(img)
        # compute() returns (refined_keypoints, descriptors)
        _, descs = describer.compute(gray, kps)
        descriptors_per_frame.append(descs)
        print(f"    frame {i:02d}: {descs.shape[0]} descriptors  shape={descs.shape}")

    return descriptors_per_frame


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #

def _build_describer() -> cv2.SIFT:
    # Same parameters as detection so keypoint refinement stays consistent
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