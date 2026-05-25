"""
Visual Hull — space carving reconstruction.

Builds a voxel grid centred on the world origin and carves away voxels
that project outside the object mask in any camera view.

Masking uses the same ellipse + colour logic as feature_detection.py so
the silhouettes are consistent with the rest of the pipeline.
"""
import cv2
import numpy as np
import os
import config


def compute_visual_hull(images, projections, grid_resolution=80):
    """
    Space carving: voxel grid carved by silhouette masks from all views.
    Returns (N, 3) array of surviving voxel centres.
    """
    masks = [_build_mask(img) for img in images]

    # Save masks for debugging
    os.makedirs("output/silhouettes", exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, masks)):
        cv2.imwrite(f"output/silhouettes/mask_{i:02d}.png", mask)
        debug = img.copy()
        debug[mask == 0] = (0, 0, 80)
        cv2.imwrite(f"output/silhouettes/debug_{i:02d}.jpg", debug)

    # Voxel grid centred on world origin
    r    = 0.07   # ±7cm in XY — generous for a 57mm cube
    z_lo = -0.02  # slightly below table surface
    z_hi =  0.12  # above top of cube
    coords_xy = np.linspace(-r, r, grid_resolution)
    coords_z  = np.linspace(z_lo, z_hi, grid_resolution)
    xs, ys, zs = np.meshgrid(coords_xy, coords_xy, coords_z)
    voxels = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    keep = np.ones(len(voxels), dtype=bool)
    X_h  = np.hstack([voxels, np.ones((len(voxels), 1))])

    for frame_idx, (P, mask) in enumerate(zip(projections, masks)):
        h, w  = mask.shape
        proj  = (P @ X_h.T).T
        depth = proj[:, 2]
        valid = depth > 0
        safe  = np.where(valid, depth, 1.0)
        px = (proj[:, 0] / safe).astype(int)
        py = (proj[:, 1] / safe).astype(int)
        in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        in_mask   = np.zeros(len(voxels), dtype=bool)
        idx = np.where(in_bounds)[0]
        if len(idx):
            in_mask[idx] = mask[py[idx], px[idx]] > 0
        if frame_idx == 0:
            sample = slice(0, 10)
            print(f"  px range: {px.min()}..{px.max()}, py range: {py.min()}..{py.max()}")
            print(f"  depth range: {depth.min():.4f}..{depth.max():.4f}")
            print(f"  in_bounds count: {in_bounds.sum()}")
            print(f"  in_mask count: {in_mask.sum()}")
        # Carve: remove voxels visible from this camera but outside its mask
        keep &= (~in_bounds | in_mask)
        print(f"    frame {frame_idx:02d}: {keep.sum()} voxels remaining")

    print(f"    Visual hull: {keep.sum()} / {len(voxels)} voxels survive")
    return voxels[keep]


# --------------------------------------------------------------------------- #
#  Mask generation — identical logic to feature_detection.py                  #
# --------------------------------------------------------------------------- #

def _build_mask(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    mask = _centre_mask(gray)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coloured = cv2.inRange(hsv, (0,  60,  60), (180, 255, 255))
    white    = cv2.inRange(hsv, (0,   0, 180), (180,  60, 255))
    cube_px  = cv2.bitwise_or(coloured, white)
    
    # --- ADD THIS ---
    print(f"    mask ellipse px: {mask.sum()//255}, "
          f"coloured px: {coloured.sum()//255}, "
          f"white px: {white.sum()//255}, "
          f"cube_px before ellipse: {cube_px.sum()//255}")
    # ----------------
    
    cube_px = cv2.bitwise_and(cube_px, mask)
    
    # ... rest unchanged
    kernel = np.ones((25, 25), np.uint8)
    cube_px = cv2.dilate(cube_px, kernel, iterations=2)
    cube_px = cv2.morphologyEx(cube_px, cv2.MORPH_CLOSE,
                                np.ones((40, 40), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cube_px)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cube_px = (labels == largest).astype(np.uint8) * 255
    
    # --- ADD THIS ---
    print(f"    final mask coverage: {cube_px.sum()//255} px "
          f"({100*cube_px.sum()/255/cube_px.size:.1f}%)")
    # ----------------
    
    return cube_px

def _centre_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx   = int((w // 2) * 1.05)
    cy   = int(h * 0.47)
    axes = (int(w * 0.2), int(h * 0.25))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)