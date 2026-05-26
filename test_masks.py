import sys, os, glob, cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "base", "src"))
from pipeline.depth_map_fusion import _compute_average_ellipse, _project_disk_mask

FRAMES_DIR = os.path.expanduser("~/Desktop/frames")
OUT_DIR = os.path.expanduser("~/Desktop/test")
os.makedirs(OUT_DIR, exist_ok=True)

paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
print(f"Found {len(paths)} frames in {FRAMES_DIR}\n")

images = []
for path in paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)

plate_ellipse = _compute_average_ellipse(images)

for i, img in enumerate(images):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, plate_ellipse, 255, -1)

    mask_px = int(mask.sum() // 255)
    print(f"  [mask {i:02d}] plate+cube: {mask_px} px ({100 * mask_px / (h * w):.1f}%)")

    cv2.imwrite(os.path.join(OUT_DIR, f"mask_{i:02d}.png"), mask)
    debug = img.copy()
    debug[mask == 0] = (debug[mask == 0] * 0.3).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(debug, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f"debug_{i:02d}.jpg"), debug)

print(f"\nDone. Results in {OUT_DIR}")
