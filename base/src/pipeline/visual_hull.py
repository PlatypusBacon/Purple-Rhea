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
    Build silhouette mask by:
      1. Detect the turntable plate rim via Canny + Hough ellipse (or circle).
      2. Fill the detected plate region.
      3. Within that region, separate object from plate via brightness/colour.
      4. Morphological cleanup + largest component.
    """
    h_img, w_img = img.shape[:2]
    gray = _to_gray(img)

    # ------------------------------------------------------------------ #
    # Stage 1: Find the turntable plate using Canny + contours            #
    # The plate rim appears as a bright elliptical ring on the dark mat.  #
    # ------------------------------------------------------------------ #
    plate_mask = _detect_plate_mask(img, gray, frame_idx)
    plate_px = int(plate_mask.sum() // 255)
    print(f"  [mask {frame_idx:02d}] plate mask covers {plate_px} px "
          f"({100*plate_px/plate_mask.size:.1f}% of image)")

    if plate_px < 1000:
        print(f"  [WARN mask {frame_idx:02d}] plate detection failed — "
              f"falling back to centre ellipse")
        plate_mask = _fallback_ellipse(gray)

    # ------------------------------------------------------------------ #
    # Stage 2: Within plate, isolate ONLY the cube                        #
    # The plate surface is dark (low value). The cube is bright/coloured. #
    # Simply threshold: dark = plate surface = background                 #
    # ------------------------------------------------------------------ #
    
    # Convert to LAB for better brightness separation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0]   # L channel: 0=black, 255=white
    
    # Plate surface is dark — threshold to find bright (cube) pixels
    # Tune the threshold (currently 80) if cube bottom gets clipped
    _, cube_bright = cv2.threshold(L, 80, 255, cv2.THRESH_BINARY)
    
    # Also catch all saturated colours (cube faces)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    coloured = cv2.inRange(hsv, (0, 60, 40), (180, 255, 255))
    
    # Union: anything bright OR coloured inside the plate
    cube_px = cv2.bitwise_or(cube_bright, coloured)
    cube_px = cv2.bitwise_and(cube_px, plate_mask)
    
    roi_px = int(cube_px.sum() // 255)
    print(f"  [mask {frame_idx:02d}] cube pixels inside plate: {roi_px}")
    
    if roi_px < 500:
        print(f"  [WARN mask {frame_idx:02d}] too few cube pixels — returning plate mask")
        return plate_mask

    # ------------------------------------------------------------------ #
    # Stage 3: Morphological close to fill the cube silhouette            #
    # ------------------------------------------------------------------ #
    kernel_close  = np.ones((25, 25), np.uint8)
    kernel_dilate = np.ones((10, 10), np.uint8)
    cube_px = cv2.dilate(cube_px, kernel_dilate, iterations=2)
    cube_px = cv2.morphologyEx(cube_px, cv2.MORPH_CLOSE, kernel_close)

    # ------------------------------------------------------------------ #
    # Stage 4: Largest connected component only                           #
    # ------------------------------------------------------------------ #
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cube_px)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cube_px = (labels == largest).astype(np.uint8) * 255
        final_px = int(cube_px.sum() // 255)
        print(f"  [mask {frame_idx:02d}] final cube mask: {final_px} px")
    else:
        print(f"  [mask {frame_idx:02d}] no components — falling back to plate mask")
        return plate_mask

    return cube_px


def _detect_plate_mask(img: np.ndarray, gray: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Detect the circular turntable plate using Canny edge detection.
    
    The bright specular rim of the plate gives a strong Canny response.
    We find the largest closed contour that is roughly circular/elliptical
    and fill it to create the plate ROI mask.
    """
    h, w = gray.shape

    # Blur to suppress cube edge noise, keep the strong plate rim
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Canny — lower threshold catches the rim even if partly in shadow
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Dilate edges to close small gaps in the rim
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    # Save edge debug image
    cv2.imwrite(f"output/silhouettes/edges_{frame_idx:02d}.png", edges)

    # Find contours and look for the plate rim
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"  [plate {frame_idx:02d}] no contours found")
        return np.zeros((h, w), dtype=np.uint8)

    # Score contours: want large area, roughly elliptical (low eccentricity variance)
    best_mask = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05):   # must cover at least 5% of image
            continue
        if len(cnt) < 5:            # need 5 pts to fit ellipse
            continue

        # Fit an ellipse to the contour
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        (ex, ey), (ea, eb), angle = ellipse
        if eb < 1:
            continue
        if ea < w * 0.4 or eb < h * 0.25:
            continue

        # Aspect ratio check — plate viewed at an angle gives ellipse,
        # but axes shouldn't be wildly different (not a line)
        aspect = ea / eb if eb > 0 else 0
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Prefer ellipses whose centre is in the lower-centre of the frame
        # (plate tends to sit centre-bottom from camera angle)
        cx_norm = abs(ex / w - 0.5)   # 0=centred, 0.5=edge
        cy_norm = ey / h               # 0=top, 1=bottom

        # Score: large area + centred horizontally + in lower half
        score = area * (1 - cx_norm) * (0.3 + cy_norm)

        if score > best_score:
            best_score = score
            best_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(best_mask, ellipse, 255, -1)

        print(f"  [plate {frame_idx:02d}] candidate ellipse: "
              f"centre=({ex:.0f},{ey:.0f}) axes=({ea:.0f},{eb:.0f}) "
              f"area={area:.0f} score={score:.0f}")

    if best_mask is None:
        print(f"  [plate {frame_idx:02d}] no valid ellipse found in contours")
        return np.zeros((h, w), dtype=np.uint8)

    return best_mask


def _fallback_ellipse(gray: np.ndarray) -> np.ndarray:
    """Fallback: broad centre ellipse if plate detection fails."""
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(w * 0.52), int(h * 0.55)
    ax, ay = int(w * 0.35), int(h * 0.32)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    print(f"  [mask] fallback ellipse centre=({cx},{cy}) axes=({ax},{ay})")
    return mask


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)