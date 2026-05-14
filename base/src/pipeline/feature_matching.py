"""
Feature Matching — cross-frame descriptor comparison.

Matches SIFT descriptors between frame pairs using a FLANN-based nearest
neighbour search with Lowe's ratio test to discard ambiguous matches.

Strategy: match every frame against its neighbours within a window, plus
wrap-around pairs (frame 0 vs frame 35 etc.) for full 360° coverage.

Input:
    descriptors_per_frame: list[np.ndarray | None]  from feature_description

Output:
    matches: dict mapping (frame_i, frame_j) -> list[cv2.DMatch]
             Only pairs with enough inliers are kept.
"""

import cv2
import numpy as np
import config

# Minimum good matches required to keep a pair
MIN_MATCH_COUNT = 8

# Lowe ratio test threshold (lower = stricter)
RATIO_THRESHOLD = 0.75

# How many neighbouring frames to match against (each side)
NEIGHBOUR_WINDOW = 2


def match_all_pairs(
    descriptors_per_frame: list[np.ndarray | None],
) -> dict[tuple[int, int], list[cv2.DMatch]]:
    """
    Match descriptors between frame pairs.
    Returns a dict: {(i, j): [DMatch, ...]}  where i < j.
    """
    matcher = _build_matcher()
    n = len(descriptors_per_frame)
    pairs = _pairs_to_match(n)
    results = {}

    for i, j in pairs:
        descs_i = descriptors_per_frame[i]
        descs_j = descriptors_per_frame[j]

        if descs_i is None or descs_j is None:
            continue
        if len(descs_i) < 2 or len(descs_j) < 2:
            continue

        good = _match_pair(matcher, descs_i, descs_j)

        if len(good) >= MIN_MATCH_COUNT:
            results[(i, j)] = good
            print(f"    frames ({i:02d},{j:02d}): {len(good)} matches")
        else:
            print(f"    frames ({i:02d},{j:02d}): {len(good)} matches — below threshold, skipped")

    return results


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #

def _build_matcher() -> cv2.FlannBasedMatcher:
    # FLANN with KD-Tree index — fast for SIFT's 128-D float descriptors
    index_params  = dict(algorithm=1, trees=5)   # algorithm=1 -> FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _match_pair(
    matcher: cv2.FlannBasedMatcher,
    descs_a: np.ndarray,
    descs_b: np.ndarray,
) -> list[cv2.DMatch]:
    """kNN match (k=2) then Lowe ratio test."""
    raw = matcher.knnMatch(descs_a.astype(np.float32),
                           descs_b.astype(np.float32), k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESHOLD * n.distance:
                good.append(m)
    return good


def _pairs_to_match(n: int) -> list[tuple[int, int]]:
    """
    Build list of (i, j) pairs to match.
    Each frame is matched against NEIGHBOUR_WINDOW frames ahead,
    with wrap-around for the end of the orbit.
    """
    pairs = set()
    for i in range(n):
        for offset in range(1, NEIGHBOUR_WINDOW + 1):
            j = (i + offset) % n
            pair = (min(i, j), max(i, j))
            # Avoid wrapping small-index pairs being added as (0, n-1)
            # — keep them only if they are genuinely close in angle
            if abs(i - j) <= NEIGHBOUR_WINDOW or abs(i - j) >= n - NEIGHBOUR_WINDOW:
                pairs.add(pair)
    return sorted(pairs)