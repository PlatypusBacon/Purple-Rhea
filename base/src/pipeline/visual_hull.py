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
import math

# Minimum number of views that must carve a voxel for it to be removed.
# Setting this to 1 (original behaviour) means any single bad mask can delete
# a voxel. Setting to 2–3 makes the hull more robust to mask noise.
MIN_CARVE_VIEWS = 1   # increase to 2 or 3 if you get Swiss-cheese artefacts


def compute_visual_hull(images, projections, grid_resolution=80):
    print(f"\n[hull] Building masks for {len(images)} images...")
    masks = [_build_mask(img, P, idx) 
             for idx, (img, P) in enumerate(zip(images, projections))]

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
        valid = depth > 0.02
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
              f"px_range=[{px[valid].min()}..{px[valid].max()}]  "
              f"py_range=[{py[valid].min()}..{py[valid].max()}]  "
              f"depth_range=[{depth[valid].min():.4f}..{depth[valid].max():.4f}]")

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

def _project_plate_mask(img: np.ndarray, P: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Project the known physical plate circle into the image using P,
    and fill it to create the plate ROI mask.
    
    Much more reliable than Canny-based detection since it uses known geometry.
    """
    h, w = img.shape[:2]
    
    plate_radius = config.RIG_BASE_LENGTH  # physical radius in metres, e.g. 0.15
    n_points = 360
    
    # Sample points around the plate circle at Z=0 (plate sits on turntable surface)
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle_3d = np.array([
        [plate_radius * np.cos(a), plate_radius * np.sin(a), 0.0, 1.0]
        for a in angles
    ])  # (360, 4)
    
    # Project into image
    proj = (P @ circle_3d.T).T  # (360, 3)
    depth = proj[:, 2]
    
    valid = depth > 0
    if valid.sum() < 10:
        print(f"  [plate {frame_idx:02d}] projected plate mostly behind camera — fallback")
        return _fallback_ellipse(img)
    
    px = (proj[:, 0] / np.where(valid, depth, 1)).astype(int)
    py = (proj[:, 1] / np.where(valid, depth, 1)).astype(int)
    
    points = np.stack([px[valid], py[valid]], axis=1)
    
    # Clamp to image bounds for drawing
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    
    px_count = int(mask.sum() // 255)
    print(f"  [plate {frame_idx:02d}] projected plate mask: {px_count} px "
          f"({100*px_count/mask.size:.1f}%)")
    
    return mask

def _build_plate_mask(img: np.ndarray, P: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Fuse projected plate geometry with white-rim detection to get a clean plate ROI.
    
    1. Project physical plate circle → expected centre + radius in pixels
    2. Search for bright rim pixels in an annular band around the projection
    3. Fit ellipse to rim inliers (if enough found)
    4. Fill solid ellipse as the plate mask
    5. Fall back to projected circle if rim detection fails
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val_ch = hsv[:, :, 2]
    sat_ch = hsv[:, :, 1]

    # ------------------------------------------------------------------ #
    # STEP 1: Project the physical plate circle to get prior              #
    # ------------------------------------------------------------------ #
    plate_radius = config.RIG_BASE_LENGTH  # physical radius in metres
    n_pts = 360
    angles = np.linspace(0, 2 * np.pi, n_pts)
    circle_3d = np.array([
        [plate_radius * np.cos(a), plate_radius * np.sin(a), 0.0, 1.0]
        for a in angles
    ])  # (360, 4)

    proj = (P @ circle_3d.T).T   # (360, 3)
    depth = proj[:, 2]
    valid = depth > 0

    if valid.sum() < 10:
        print(f"  [plate {frame_idx:02d}] projection mostly behind camera — pure fallback ellipse")
        return _fallback_ellipse(img)

    safe_depth = np.where(valid, depth, 1.0)
    px_proj = (proj[:, 0] / safe_depth)
    py_proj = (proj[:, 1] / safe_depth)

    # Estimate projected centre and radius from the projected points
    valid_px = px_proj[valid]
    valid_py = py_proj[valid]
    prior_cx = float(np.mean(valid_px))
    prior_cy = float(np.mean(valid_py))
    # Radius = mean distance of projected rim points from projected centre
    dists = np.sqrt((valid_px - prior_cx)**2 + (valid_py - prior_cy)**2)
    prior_r = float(np.mean(dists))

    print(f"  [plate {frame_idx:02d}] projected prior: "
          f"centre=({prior_cx:.1f},{prior_cy:.1f}) r={prior_r:.1f}px")

    # ------------------------------------------------------------------ #
    # STEP 2: Find white rim pixels in annular band around prior          #
    # The physical rim is bright (high val) and low saturation            #
    # Search band: prior_r * [0.70, 1.30]                                 #
    # ------------------------------------------------------------------ #
    band_inner = prior_r * 0.70
    band_outer = prior_r * 1.30

    # Distance from prior centre for every pixel
    yy, xx = np.mgrid[0:h, 0:w]
    dist_from_prior = np.sqrt((xx - prior_cx)**2 + (yy - prior_cy)**2)

    in_band = (dist_from_prior >= band_inner) & (dist_from_prior <= band_outer)

    # White rim: bright + desaturated
    is_rim = (val_ch > 160) & (sat_ch < 60)

    rim_pixels = in_band & is_rim

    rim_ys, rim_xs = np.where(rim_pixels)
    n_rim = len(rim_xs)
    print(f"  [plate {frame_idx:02d}] rim pixels in band: {n_rim}")

    # ------------------------------------------------------------------ #
    # STEP 3: Fit ellipse to rim inliers (need ≥ 20 pixels)              #
    # ------------------------------------------------------------------ #
    use_projection = False
    fitted_ellipse = None

    if n_rim >= 20:
        rim_points = np.column_stack([rim_xs, rim_ys]).astype(np.float32)

        try:
            # fitEllipse needs (N,1,2) contour format
            ellipse = cv2.fitEllipse(rim_points.reshape(-1, 1, 2).astype(np.int32))
            (ex, ey), (ea, eb), angle = ellipse

            # Sanity checks against prior
            fitted_r = (ea + eb) / 4.0  # mean semi-axis
            centre_offset = np.sqrt((ex - prior_cx)**2 + (ey - prior_cy)**2)
            r_ratio = fitted_r / prior_r if prior_r > 0 else 0

            print(f"  [plate {frame_idx:02d}] fitted ellipse: "
                  f"centre=({ex:.1f},{ey:.1f}) axes=({ea:.1f},{eb:.1f}) "
                  f"offset={centre_offset:.1f}px r_ratio={r_ratio:.2f}")

            if centre_offset < prior_r * 0.4 and 0.6 < r_ratio < 1.5:
                fitted_ellipse = ellipse
                print(f"  [plate {frame_idx:02d}] using FITTED ellipse")
            else:
                print(f"  [plate {frame_idx:02d}] fitted ellipse failed sanity check — using projection")
                use_projection = True
        except cv2.error as e:
            print(f"  [plate {frame_idx:02d}] fitEllipse failed ({e}) — using projection")
            use_projection = True
    else:
        print(f"  [plate {frame_idx:02d}] too few rim pixels ({n_rim}) — using projection")
        use_projection = True

    # ------------------------------------------------------------------ #
    # STEP 4: Build plate mask                                            #
    # ------------------------------------------------------------------ #
    mask = np.zeros((h, w), dtype=np.uint8)

    if fitted_ellipse is not None:
        (ex, ey), (ea, eb), angle = fitted_ellipse
        # Add 5% margin so we don't clip the plate edge
        cv2.ellipse(mask, (int(ex), int(ey)),
                    (int(ea * 0.55), int(eb * 0.55)),
                    angle, 0, 360, 255, -1)
    else:
        # Fall back: use projected points directly — clamp to image
        pts = np.column_stack([
            np.clip(valid_px, 0, w - 1),
            np.clip(valid_py, 0, h - 1)
        ]).astype(np.int32)

        if len(pts) >= 5:
            try:
                ellipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
                (ex, ey), (ea, eb), angle = ellipse
                cv2.ellipse(mask, (int(ex), int(ey)),
                            (int(ea * 0.55), int(eb * 0.55)),
                            angle, 0, 360, 255, -1)
                print(f"  [plate {frame_idx:02d}] projection fallback ellipse: "
                      f"centre=({ex:.1f},{ey:.1f}) axes=({ea:.1f},{eb:.1f})")
            except cv2.error:
                # Last resort: draw circle at prior centre/radius
                cv2.circle(mask, (int(prior_cx), int(prior_cy)), int(prior_r * 1.05), 255, -1)
                print(f"  [plate {frame_idx:02d}] last resort: circle at prior")
        else:
            cv2.circle(mask, (int(prior_cx), int(prior_cy)), int(prior_r * 1.05), 255, -1)

    px_count = int((mask > 0).sum())
    print(f"  [plate {frame_idx:02d}] plate mask: {px_count}px ({100*px_count/mask.size:.1f}%)")

    if frame_idx < 6 or frame_idx % 6 == 0:
        os.makedirs("output/silhouettes", exist_ok=True)
        # Visualise: prior circle + rim pixels + final mask
        vis = img.copy()
        vis[rim_pixels] = (0, 255, 255)          # yellow = rim pixels found
        cv2.circle(vis, (int(prior_cx), int(prior_cy)),
                   int(prior_r), (0, 255, 0), 2)  # green = projected prior
        cv2.circle(vis, (int(prior_cx), int(prior_cy)),
                   int(band_inner), (128, 128, 0), 1)
        cv2.circle(vis, (int(prior_cx), int(prior_cy)),
                   int(band_outer), (128, 128, 0), 1)
        contour_mask = mask.copy()
        vis[contour_mask == 0] = (vis[contour_mask == 0] * 0.4).astype(np.uint8)
        cv2.imwrite(f"output/silhouettes/plate_vis_{frame_idx:02d}.jpg", vis)
        cv2.imwrite(f"output/silhouettes/plate_mask_{frame_idx:02d}.png", mask)

    return mask


def _build_mask(img: np.ndarray, P: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val_ch = hsv[:, :, 2]
    sat_ch = hsv[:, :, 1]

    # Get plate ROI using fused projection + rim detection
    plate_mask = _build_plate_mask(img, P, frame_idx)

    # ------------------------------------------------------------------ #
    # Detect cube pixels within plate ROI                                 #
    # Cube: saturated stickers OR bright white stickers                   #
    # ------------------------------------------------------------------ #
    high_sat   = (sat_ch > 70).astype(np.uint8) * 255
    bright_white = ((val_ch > 160) & (sat_ch < 60)).astype(np.uint8) * 255
    cube_raw   = cv2.bitwise_or(high_sat, bright_white)
    cube_in_roi = cv2.bitwise_and(cube_raw, plate_mask)

    # Close to bridge black grid lines between stickers
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    cube_closed = cv2.morphologyEx(cube_in_roi, cv2.MORPH_CLOSE, close_k)

    # Open to remove isolated speckle
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cube_clean = cv2.morphologyEx(cube_closed, cv2.MORPH_OPEN, open_k)

    # Keep largest connected component(s) only
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cube_clean, connectivity=8)
    final_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = float(np.max(areas))
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= largest_area * 0.15:
                final_mask[labels == lbl] = 255

        merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, merge_k)

        px_count = int((final_mask > 0).sum())
        print(f"  [mask {frame_idx:02d}] final cube mask: {px_count}px "
              f"({100*px_count/final_mask.size:.1f}%)")
    else:
        print(f"  [mask {frame_idx:02d}] WARNING: no cube pixels found in plate ROI")

    if frame_idx < 6 or frame_idx % 6 == 0:
        os.makedirs("output/silhouettes", exist_ok=True)
        cv2.imwrite(f"output/silhouettes/final_mask_{frame_idx:02d}.png", final_mask)
        debug = img.copy()
        debug[final_mask == 0] = (0, 0, 80)
        cv2.imwrite(f"output/silhouettes/debug_{frame_idx:02d}.jpg", debug)

    return final_mask


def _detect_plate_mask(img: np.ndarray, gray: np.ndarray, frame_idx: int) -> np.ndarray:
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, threshold1=20, threshold2=80)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    cv2.imwrite(f"output/silhouettes/edges_{frame_idx:02d}.png", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  [plate {frame_idx:02d}] no contours found")
        return np.zeros((h, w), dtype=np.uint8)

    best_mask = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Lower area threshold — partial arcs still enclose significant area
        if area < (h * w * 0.02):
            continue
        if len(cnt) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        (ex, ey), (ea, eb), angle = ellipse
        # ea >= eb always (OpenCV convention: ea is major axis)
        if eb < 1:
            continue

        # Accept ellipses where at least ONE axis covers ~30% of the image
        # This handles partial plate rims that extend beyond the frame
        if ea < w * 0.25 and eb < h * 0.20:
            continue

        aspect = ea / eb if eb > 0 else 999
        # Allow more elongated ellipses (low-angle shots compress the plate)
        if aspect < 0.15 or aspect > 8.0:
            continue

        # Circularity of the contour itself (not the fitted ellipse)
        # Real plate rim arc has high circularity; rig clutter does not
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)

        # Centre should be roughly in the image (allow outside for partial plates)
        cx_norm = abs(ex / w - 0.5)

        # Heavily weight circularity to reject rig/clutter contours
        score = area * (circularity ** 2) * (1.2 - cx_norm)

        print(f"  [plate {frame_idx:02d}] candidate: "
              f"centre=({ex:.0f},{ey:.0f}) axes=({ea:.0f},{eb:.0f}) "
              f"circ={circularity:.3f} area={area:.0f} score={score:.0f}")

        if score > best_score:
            best_score = score
            best_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(best_mask, ellipse, 255, -1)

    if best_mask is None:
        print(f"  [plate {frame_idx:02d}] no valid ellipse found")
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