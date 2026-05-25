"""
calibrate_fx.py — Grid-search focal length calibration.

Runs triangulation across all matched frame pairs for a range of fx values,
scoring each by the number of points that survive a tight reprojection filter.
The fx with the highest survivor count is your best estimate.

Usage:
    python calibrate_fx.py

    Optionally pass a range override:
        python calibrate_fx.py --fmin 600 --fmax 2200 --step 50

Hook-up:
    Comment out everything in orchestrator.py after Step 4 (feature matching)
    so the pipeline returns the `matches` and `keypoints_per_frame` objects,
    then call this script — or just run it standalone if you save those objects
    to disk (see bottom of this file for a save/load helper).

The script can also be imported and called directly:
    from calibrate_fx import find_best_fx
    best_fx, results = find_best_fx(matches, keypoints_per_frame, projections_at_unit_fx)
"""

import argparse
import math
import numpy as np
import cv2
import os
import sys

# ---------------------------------------------------------------------------
# Configuration — edit to match your rig
# ---------------------------------------------------------------------------
IMAGE_WIDTH  = 1600
IMAGE_HEIGHT = 1200

# Search range for fx (pixels).  OV2640 at 1600×1200 is typically 1200–2000.
FX_MIN  = 600
FX_MAX  = 2400
FX_STEP = 50

# Reprojection error threshold (pixels) — a point "survives" if both source
# frames reproject it within this many pixels.
REPROJ_THRESHOLD = 3.0

# Minimum matches per pair to bother triangulating
MIN_MATCHES = 8

# Where your pipeline saves its intermediate .npy files
OUTPUT_DIR = "output"


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def _make_K(fx: float) -> np.ndarray:
    cx = IMAGE_WIDTH  / 2.0
    cy = IMAGE_HEIGHT / 2.0
    return np.array([[fx, 0, cx],
                     [0, fx, cy],
                     [0,  0,  1]], dtype=np.float64)


def _reproject_error(P: np.ndarray, X3d: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Return per-point reprojection error (pixels)."""
    X4 = np.hstack([X3d, np.ones((len(X3d), 1))]).T   # 4×N
    proj = P @ X4                                       # 3×N
    proj = (proj[:2] / proj[2]).T                       # N×2
    return np.linalg.norm(proj - pts2d, axis=1)


def _poses_for_fx(fx: float, base_projections: list[np.ndarray],
                  base_K: np.ndarray) -> list[np.ndarray]:
    """
    Re-express projections under a new K.
    base_projections were built with base_K; we extract [R|t] and rebuild.
    """
    new_K = _make_K(fx)
    base_K_inv = np.linalg.inv(base_K)
    new_projections = []
    for P in base_projections:
        Rt = base_K_inv @ P          # recover [R|t]
        new_projections.append(new_K @ Rt)
    return new_projections


def score_fx(fx: float,
             matches: dict,
             keypoints_per_frame: list,
             base_projections: list[np.ndarray],
             base_K: np.ndarray,
             reproj_threshold: float = REPROJ_THRESHOLD) -> dict:
    """
    Triangulate all matched pairs under `fx` and count survivors.

    Returns a dict with keys:
        fx, survivors, total_triangulated, pairs_with_survivors,
        mean_error_survivors, median_error_survivors
    """
    projections = _poses_for_fx(fx, base_projections, base_K)

    total_triangulated = 0
    survivors          = 0
    pair_survivor_list = []
    all_errors         = []

    for (i, j), dmatches in matches.items():
        if len(dmatches) < MIN_MATCHES:
            continue

        kps_i = keypoints_per_frame[i]
        kps_j = keypoints_per_frame[j]

        pts_i = np.float32([kps_i[m.queryIdx].pt for m in dmatches])
        pts_j = np.float32([kps_j[m.trainIdx].pt for m in dmatches])

        Pi = projections[i]
        Pj = projections[j]

        X4d = cv2.triangulatePoints(Pi, Pj, pts_i.T, pts_j.T)
        # Reject points behind either camera (w < 0)
        w = X4d[3]
        valid = w > 1e-6
        if valid.sum() == 0:
            continue

        X3d = (X4d[:3, valid] / w[valid]).T   # (M, 3)
        pts_i_v = pts_i[valid]
        pts_j_v = pts_j[valid]

        err_i = _reproject_error(Pi, X3d, pts_i_v)
        err_j = _reproject_error(Pj, X3d, pts_j_v)

        keep = (err_i < reproj_threshold) & (err_j < reproj_threshold)
        n_kept = keep.sum()

        total_triangulated += len(X3d)
        survivors          += n_kept
        if n_kept > 0:
            pair_survivor_list.append((i, j, n_kept))
            all_errors.extend(err_i[keep].tolist())
            all_errors.extend(err_j[keep].tolist())

    mean_err   = float(np.mean(all_errors))   if all_errors else float("inf")
    median_err = float(np.median(all_errors)) if all_errors else float("inf")

    return {
        "fx":                    fx,
        "survivors":             survivors,
        "total_triangulated":    total_triangulated,
        "pairs_with_survivors":  len(pair_survivor_list),
        "mean_error_survivors":  mean_err,
        "median_error_survivors": median_err,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def find_best_fx(matches: dict,
                 keypoints_per_frame: list,
                 base_projections: list[np.ndarray],
                 fx_min:  float = FX_MIN,
                 fx_max:  float = FX_MAX,
                 fx_step: float = FX_STEP,
                 reproj_threshold: float = REPROJ_THRESHOLD,
                 refine: bool = True) -> tuple[float, list[dict]]:
    """
    Grid-search for the best fx, then optionally refine with a finer grid
    around the coarse winner.

    Returns (best_fx, all_results_sorted_by_survivors).
    """
    base_K = _make_K(fx_min)   # arbitrary reference — we just need a K to
                                # extract Rt from the existing projections

    fx_values = np.arange(fx_min, fx_max + fx_step, fx_step)
    results   = []

    print(f"\nCoarse grid search: fx = {fx_min} → {fx_max}, step {fx_step}")
    print(f"{'fx':>8}  {'survivors':>10}  {'pairs':>6}  "
          f"{'median_err':>10}  {'mean_err':>10}")
    print("-" * 52)

    for fx in fx_values:
        r = score_fx(fx, matches, keypoints_per_frame,
                     base_projections, base_K, reproj_threshold)
        results.append(r)
        print(f"{fx:8.0f}  {r['survivors']:10d}  {r['pairs_with_survivors']:6d}  "
              f"{r['median_error_survivors']:10.3f}  {r['mean_error_survivors']:10.3f}")

    best_coarse = max(results, key=lambda r: r["survivors"])
    print(f"\nCoarse best: fx = {best_coarse['fx']:.0f}  "
          f"({best_coarse['survivors']} survivors)")

    if not refine:
        results_sorted = sorted(results, key=lambda r: -r["survivors"])
        return best_coarse["fx"], results_sorted

    # --- Fine grid around coarse winner ---
    fine_step   = fx_step / 10.0
    fine_centre = best_coarse["fx"]
    fine_min    = max(fx_min, fine_centre - fx_step)
    fine_max    = min(fx_max, fine_centre + fx_step)
    fine_values = np.arange(fine_min, fine_max + fine_step, fine_step)

    print(f"\nFine grid search: fx = {fine_min:.0f} → {fine_max:.0f}, "
          f"step {fine_step:.1f}")
    print(f"{'fx':>8}  {'survivors':>10}  {'pairs':>6}  "
          f"{'median_err':>10}  {'mean_err':>10}")
    print("-" * 52)

    fine_results = []
    for fx in fine_values:
        r = score_fx(fx, matches, keypoints_per_frame,
                     base_projections, base_K, reproj_threshold)
        fine_results.append(r)
        print(f"{fx:8.1f}  {r['survivors']:10d}  {r['pairs_with_survivors']:6d}  "
              f"{r['median_error_survivors']:10.3f}  {r['mean_error_survivors']:10.3f}")

    all_results   = results + fine_results
    results_sorted = sorted(all_results, key=lambda r: -r["survivors"])
    best           = results_sorted[0]

    print(f"\n{'='*52}")
    print(f"  BEST fx = {best['fx']:.1f} px")
    print(f"  survivors          : {best['survivors']}")
    print(f"  pairs contributing : {best['pairs_with_survivors']}")
    print(f"  median reproj err  : {best['median_error_survivors']:.3f} px")
    print(f"  mean   reproj err  : {best['mean_error_survivors']:.3f} px")
    print(f"{'='*52}")
    print(f"\nPaste this into pose_computation.py → _camera_intrinsics():")
    print(f"    fx = {best['fx']:.1f}")

    return best["fx"], results_sorted


# ---------------------------------------------------------------------------
# Save / load helpers so you don't re-run the full pipeline each time
# ---------------------------------------------------------------------------

def save_calibration_inputs(matches, keypoints_per_frame, projections,
                             directory: str = OUTPUT_DIR):
    import pickle
    os.makedirs(directory, exist_ok=True)

    # Serialise DMatch objects as plain tuples (queryIdx, trainIdx, distance)
    matches_serialisable = {
        key: [(m.queryIdx, m.trainIdx, m.distance) for m in dmatches]
        for key, dmatches in matches.items()
    }

    # Serialise KeyPoints as plain tuples (x, y)
    kps_serialisable = [
        [(kp.pt[0], kp.pt[1]) for kp in kps]
        for kps in keypoints_per_frame
    ]

    path = os.path.join(directory, "calib_inputs.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "matches":             matches_serialisable,
            "keypoints_per_frame": kps_serialisable,
            "projections":         projections,
        }, f)
    print(f"Saved calibration inputs → {path}")


def load_calibration_inputs(directory: str = OUTPUT_DIR):
    import pickle
    path = os.path.join(directory, "calib_inputs.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct DMatch objects
    matches = {}
    for key, tuples in data["matches"].items():
        dmatches = []
        for queryIdx, trainIdx, distance in tuples:
            m = cv2.DMatch()
            m.queryIdx  = queryIdx
            m.trainIdx  = trainIdx
            m.distance  = distance
            dmatches.append(m)
        matches[key] = dmatches

    # Reconstruct KeyPoints (only .pt is needed for triangulation)
    keypoints_per_frame = [
        [cv2.KeyPoint(x=x, y=y, size=1) for x, y in kps]
        for kps in data["keypoints_per_frame"]
    ]

    return matches, keypoints_per_frame, data["projections"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Grid-search focal length calibration")
    p.add_argument("--fmin",      type=float, default=FX_MIN)
    p.add_argument("--fmax",      type=float, default=FX_MAX)
    p.add_argument("--step",      type=float, default=FX_STEP)
    p.add_argument("--threshold", type=float, default=REPROJ_THRESHOLD,
                   help="Reprojection error threshold in pixels")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip fine grid around coarse winner")
    p.add_argument("--inputs",    type=str, default=OUTPUT_DIR,
                   help="Directory containing calib_inputs.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("Loading calibration inputs from disk...")
    try:
        matches, keypoints_per_frame, projections = load_calibration_inputs(args.inputs)
    except FileNotFoundError:
        print(f"\nERROR: could not find calib_inputs.pkl in '{args.inputs}'")
        print("Add this call near the end of your pipeline (after feature matching):")
        print("    from calibrate_fx import save_calibration_inputs")
        print("    save_calibration_inputs(matches, keypoints_per_frame, projections)")
        sys.exit(1)

    best_fx, results = find_best_fx(
        matches, keypoints_per_frame, projections,
        fx_min=args.fmin,
        fx_max=args.fmax,
        fx_step=args.step,
        reproj_threshold=args.threshold,
        refine=not args.no_refine,
    )