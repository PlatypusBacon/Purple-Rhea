"""
Visual Hull — space carving reconstruction.

Builds a voxel grid centred on the world origin and carves away voxels
that project outside the object mask in any camera view.

FIXES applied vs original:
  - Voxel grid bounds now read from config (was hardcoded to r=0.07, z_hi=0.12
    which conflicts with config.VOXEL_XY_EXTENT=0.12, VOXEL_Z_MAX=0.22).
  - Mask ellipse size/position made configurable and prints actual pixel coverage
    so you can see immediately if the mask is missing the object.
  - Added voting threshold: voxels carved by fewer than MIN_CARVE_VIEWS views
    are kept, which handles noisy/bad masks without wiping the entire volume.
  - Extensive debug prints at every carving step.
"""
import cv2
import numpy as np
import os
import config

# Minimum number of views that must carve a voxel for it to be removed.
# Setting this to 1 (original behaviour) means any single bad mask can delete
# a voxel. Setting to 2–3 makes the hull more robust to mask noise.
MIN_CARVE_VIEWS = 1   # increase to 2 or 3 if you get Swiss-cheese artefacts


def compute_visual_hull(images, projections, grid_resolution=80):
    """
    Space carving: voxel grid carved by silhouette masks from all views.
    Returns (N, 3) array of surviving voxel centres.
    """
    print(f"\n[hull] Building masks for {len(images)} images...")
    masks = [_build_mask(img, idx) for idx, img in enumerate(images)]

    # Save masks for debugging
    os.makedirs("output/silhouettes", exist_ok=True)
    for i, (img, mask) in enumerate(zip(images, masks)):
        cv2.imwrite(f"output/silhouettes/mask_{i:02d}.png", mask)
        debug = img.copy()
        debug[mask == 0] = (0, 0, 80)   # dark red = carved region
        cv2.imwrite(f"output/silhouettes/debug_{i:02d}.jpg", debug)

    # ------------------------------------------------------------------ #
    # FIX: read grid bounds from config instead of hardcoding             #
    # ------------------------------------------------------------------ #
    r    = config.VOXEL_XY_EXTENT   # was hardcoded 0.07 (config says 0.12)
    z_lo = config.VOXEL_Z_MIN       # was hardcoded -0.02
    z_hi = config.VOXEL_Z_MAX       # was hardcoded 0.12 (config says 0.22)

    print(f"[hull] Voxel grid: XY=±{r:.3f}m  Z=[{z_lo:.3f}, {z_hi:.3f}]m  "
          f"resolution={grid_resolution}³")

    coords_xy = np.linspace(-r, r, grid_resolution)
    coords_z  = np.linspace(z_lo, z_hi, grid_resolution)
    xs, ys, zs = np.meshgrid(coords_xy, coords_xy, coords_z)
    voxels = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    print(f"[hull] Total voxels: {len(voxels)}")

    # carve_votes[i] = number of views that want to carve voxel i
    carve_votes = np.zeros(len(voxels), dtype=np.int32)
    X_h = np.hstack([voxels, np.ones((len(voxels), 1))])

    for frame_idx, (P, mask) in enumerate(zip(projections, masks)):
        h_img, w_img = mask.shape
        proj  = (P @ X_h.T).T          # (N_voxels, 3)
        depth = proj[:, 2]

        # Only project voxels in front of camera
        valid = depth > 0
        safe  = np.where(valid, depth, 1.0)
        px = (proj[:, 0] / safe).astype(int)
        py = (proj[:, 1] / safe).astype(int)

        in_bounds = valid & (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img)
        in_mask   = np.zeros(len(voxels), dtype=bool)
        idx       = np.where(in_bounds)[0]

        if len(idx):
            in_mask[idx] = mask[py[idx], px[idx]] > 0

        # Voxels that are visible from this camera but NOT in the mask → vote to carve
        wants_to_carve = in_bounds & ~in_mask
        carve_votes[wants_to_carve] += 1

        n_in_bounds   = int(in_bounds.sum())
        n_in_mask     = int(in_mask.sum())
        n_carve_votes = int(wants_to_carve.sum())
        n_surviving   = int((carve_votes < MIN_CARVE_VIEWS).sum()) if frame_idx == 0 \
                        else int((carve_votes < MIN_CARVE_VIEWS + frame_idx).sum())

        print(f"  frame {frame_idx:02d}: "
              f"in_bounds={n_in_bounds}  in_mask={n_in_mask}  "
              f"carve_votes_added={n_carve_votes}  "
              f"px_range=[{px.min()}..{px.max()}]  "
              f"py_range=[{py.min()}..{py.max()}]  "
              f"depth_range=[{depth.min():.4f}..{depth.max():.4f}]")

        if n_in_bounds == 0:
            print(f"  [WARN] frame {frame_idx:02d}: ZERO voxels project inside image! "
                  f"Check projections and grid bounds.")
        if n_in_mask == 0 and n_in_bounds > 0:
            print(f"  [WARN] frame {frame_idx:02d}: voxels project into image but NONE "
                  f"fall inside mask — mask may be empty or misaligned.")

    # Apply voting threshold
    keep = carve_votes < MIN_CARVE_VIEWS
    surviving = int(keep.sum())

    print(f"\n[hull] Carving complete.")
    print(f"[hull] MIN_CARVE_VIEWS threshold = {MIN_CARVE_VIEWS}")
    print(f"[hull] Surviving voxels: {surviving} / {len(voxels)} "
          f"({100*surviving/len(voxels):.1f}%)")

    if surviving == len(voxels):
        print("[hull] WARNING: ALL voxels survived — nothing was carved! "
              "Possible causes:\n"
              "  1. All masks are empty (check output/silhouettes/)\n"
              "  2. No voxels project inside the image (check projections)\n"
              "  3. Camera centres at origin (radius=0 issue in pose_computation)")
    elif surviving == 0:
        print("[hull] WARNING: ZERO voxels survived — everything was carved! "
              "Possible causes:\n"
              "  1. Mask covers the whole image (check output/silhouettes/)\n"
              "  2. Grid bounds don't match object size")
    else:
        print(f"[hull] Result looks plausible — {surviving} voxels form the hull.")

    return voxels[keep]


# --------------------------------------------------------------------------- #
#  Mask generation                                                             #
# --------------------------------------------------------------------------- #

def _build_mask(img: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """
    Build a silhouette mask for the object in the image.

    Strategy:
      1. Start with a broad ellipse centred on where the object typically appears.
      2. Within that ellipse, find 'coloured' and 'white' pixels (HSV ranges).
      3. Dilate + close morphologically to fill gaps.
      4. Keep only the largest connected component.

    Debug prints report pixel counts at each stage so you can identify where
    the mask goes wrong.
    """
    h_img, w_img = img.shape[:2]
    gray = _to_gray(img)

    # ------------------------------------------------------------------ #
    # Stage 1: broad ellipse — defines the region of interest             #
    # Tweak cx_frac/cy_frac/axes_frac if the object isn't centred here.  #
    # ------------------------------------------------------------------ #
    ellipse_mask = _centre_mask(gray)
    ellipse_px   = int(ellipse_mask.sum() // 255)
    print(f"  [mask {frame_idx:02d}] ellipse covers {ellipse_px} px "
          f"({100*ellipse_px/ellipse_mask.size:.1f}% of image)")

    if ellipse_px == 0:
        print(f"  [WARN mask {frame_idx:02d}] ellipse mask is EMPTY — "
              f"check _centre_mask parameters vs image size {w_img}×{h_img}")
        return np.zeros((h_img, w_img), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Stage 2: colour detection inside ellipse                            #
    # ------------------------------------------------------------------ #
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coloured = cv2.inRange(hsv, (0,  60,  60), (180, 255, 255))
    white    = cv2.inRange(hsv, (0,   0, 180), (180,  60, 255))
    cube_px  = cv2.bitwise_or(coloured, white)

    col_px   = int(coloured.sum() // 255)
    wht_px   = int(white.sum()    // 255)
    union_px = int(cube_px.sum()  // 255)
    print(f"  [mask {frame_idx:02d}] colour detection (full image): "
          f"coloured={col_px}  white={wht_px}  union={union_px}")

    # Restrict to ellipse ROI
    cube_px = cv2.bitwise_and(cube_px, ellipse_mask)
    roi_px  = int(cube_px.sum() // 255)
    print(f"  [mask {frame_idx:02d}] after ellipse restriction: {roi_px} px")

    if roi_px == 0:
        print(f"  [WARN mask {frame_idx:02d}] NO object pixels inside ellipse! "
              f"Check HSV ranges and ellipse position.")
        # Return just the ellipse as a fallback so we don't carve everything
        print(f"  [mask {frame_idx:02d}] FALLBACK: returning ellipse mask itself")
        return ellipse_mask

    # ------------------------------------------------------------------ #
    # Stage 3: morphological close to fill gaps in the silhouette         #
    # ------------------------------------------------------------------ #
    kernel_dilate = np.ones((25, 25), np.uint8)
    kernel_close  = np.ones((40, 40), np.uint8)
    cube_px = cv2.dilate(cube_px, kernel_dilate, iterations=2)
    cube_px = cv2.morphologyEx(cube_px, cv2.MORPH_CLOSE, kernel_close)
    morph_px = int(cube_px.sum() // 255)
    print(f"  [mask {frame_idx:02d}] after morphology: {morph_px} px")

    # ------------------------------------------------------------------ #
    # Stage 4: keep only largest connected component                      #
    # ------------------------------------------------------------------ #
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cube_px)
    if n > 1:
        # stats[0] is background; find largest foreground component
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cube_px = (labels == largest).astype(np.uint8) * 255
        final_px = int(cube_px.sum() // 255)
        print(f"  [mask {frame_idx:02d}] largest component: {final_px} px "
              f"({100*final_px/cube_px.size:.1f}% of image)  "
              f"[{n-1} components found]")
    else:
        print(f"  [mask {frame_idx:02d}] no foreground components found after morphology")
        cube_px = np.zeros((h_img, w_img), dtype=np.uint8)

    return cube_px


def _centre_mask(gray: np.ndarray) -> np.ndarray:
    """
    Elliptical ROI mask.  Tweak the fractions below to match where the
    turntable / object actually appears in your frames.

    Current values:
      cx = 52.5% across  (slightly right of centre)
      cy = 47%   down    (slightly above centre)
      semi-axes: 20% width × 25% height   ← quite small; may need enlarging
    """
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    cx   = int(w * 0.525)    # horizontal centre of ellipse
    cy   = int(h * 0.47)     # vertical centre of ellipse
    # Semi-axes — increase these if the object is being clipped
    ax   = int(w * 0.30)     # was 0.20 — enlarged to cover more of the frame
    ay   = int(h * 0.35)     # was 0.25 — enlarged

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    print(f"  [mask] ellipse centre=({cx},{cy}) axes=({ax},{ay}) "
          f"image={w}×{h}")
    return mask


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)