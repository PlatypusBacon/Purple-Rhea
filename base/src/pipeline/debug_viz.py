"""
Debug Visualisation — per-frame keypoint and reprojection overlays.

Saves to output/debug/ :
  kp_frame{i:02d}.jpg        — detected keypoints on the raw image
  reproj_frame{i:02d}.jpg    — triangulated points reprojected onto each frame
                               green = inside mask, red = outside mask
"""
import cv2
import numpy as np
import os

DEBUG_DIR = "output/debug"


def save_keypoint_images(
    images: list[np.ndarray],
    keypoints_per_frame: list[list[cv2.KeyPoint]],
    masks: list[np.ndarray] | None = None,
) -> None:
    """
    For each frame: draw detected keypoints on the image.
    Keypoints inside the mask are drawn green, outside red.
    Also prints per-frame keypoint count for a quick sanity check.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for i, (img, kps) in enumerate(zip(images, keypoints_per_frame)):
        out = img.copy()
        mask = masks[i] if masks else None

        for kp in kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            h, w = out.shape[:2]
            if mask is not None and 0 <= x < w and 0 <= y < h:
                in_mask = mask[y, x] > 0
                colour = (0, 255, 0) if in_mask else (0, 0, 255)
            else:
                colour = (0, 255, 255)  # yellow = no mask available
            cv2.circle(out, (x, y), 4, colour, -1)

        # Annotate frame index and keypoint count
        label = f"Frame {i:02d} — {len(kps)} keypoints"
        cv2.putText(out, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        path = os.path.join(DEBUG_DIR, f"kp_frame{i:02d}.jpg")
        cv2.imwrite(path, out)

    print(f"  [debug] Keypoint images saved to {DEBUG_DIR}/kp_frame*.jpg")


def save_reprojection_images(
    images: list[np.ndarray],
    points_3d: np.ndarray,
    projections: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    label: str = "points",
) -> None:
    """
    Reproject a set of 3D points onto each frame and save the overlay.
    Green = reprojects inside mask, red = reprojects outside mask.
    Useful for checking projection matrix scale/alignment per frame.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)

    if len(points_3d) == 0:
        print(f"  [debug] No points to reproject for '{label}'")
        return

    X_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # (N, 4)

    for i, (img, P) in enumerate(zip(images, projections)):
        out = img.copy()
        h, w = out.shape[:2]
        mask = masks[i] if masks else None

        proj = (P @ X_h.T).T   # (N, 3)
        depth = proj[:, 2]
        valid = depth > 0

        px = np.where(valid, proj[:, 0] / np.where(valid, depth, 1), -1).astype(int)
        py = np.where(valid, proj[:, 1] / np.where(valid, depth, 1), -1).astype(int)

        in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        indices = np.where(in_bounds)[0]

        n_in_mask = 0
        n_out_mask = 0

        for idx in indices:
            x, y = px[idx], py[idx]
            if mask is not None:
                in_mask = mask[y, x] > 0
            else:
                in_mask = True
            colour = (0, 255, 0) if in_mask else (0, 0, 255)
            cv2.circle(out, (x, y), 3, colour, -1)
            if in_mask:
                n_in_mask += 1
            else:
                n_out_mask += 1

        lbl = (f"Frame {i:02d} [{label}] "
               f"in={n_in_mask} out={n_out_mask} "
               f"behind={valid.size - valid.sum()}")
        cv2.putText(out, lbl, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        path = os.path.join(DEBUG_DIR, f"reproj_{label}_frame{i:02d}.jpg")
        cv2.imwrite(path, out)
        print(f"    frame {i:02d}: {n_in_mask} in-mask  "
              f"{n_out_mask} out-of-mask  "
              f"{valid.size - valid.sum()} behind camera")

    print(f"  [debug] Reprojection images saved to {DEBUG_DIR}/reproj_{label}_*.jpg")


def load_masks(n_frames: int) -> list[np.ndarray] | None:
    """Load the lossless binary masks saved by feature_detection."""
    masks = []
    for i in range(n_frames):
        path = f"output/mask_frame{i:02d}.png"
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"  [debug] mask not found: {path} — skipping mask overlay")
            return None
        masks.append(m)
    return masks