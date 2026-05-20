import sys, os
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from comms.disk_session import start_disk_session

def load_obj(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float64)

def project_points(points_3d, P, h, w):
    X_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    proj = (P @ X_h.T).T
    depths = proj[:, 2]
    valid = depths > 0
    px = proj[valid, 0] / depths[valid]
    py = proj[valid, 1] / depths[valid]
    in_frame = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return px[in_frame], py[in_frame]

def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    raw_points      = np.load(os.path.join(config.OUTPUT_DIR, "reconstruction_raw.npy"))
    filtered_points = np.load(os.path.join(config.OUTPUT_DIR, "reconstruction_filtered.npy"))
    projections     = np.load(os.path.join(config.OUTPUT_DIR, "projections.npy"))

    print(f"Raw: {len(raw_points)}  Filtered: {len(filtered_points)}  Projections: {len(projections)}")

    session = start_disk_session()
    img = list(session)[frame_idx].load_image()
    P = projections[frame_idx]
    h, w = img.shape[:2]
    out = img.copy()

    # Red = triangulated but rejected by filter
    px, py = project_points(raw_points, P, h, w)
    for x, y in zip(px, py):
        cv2.circle(out, (int(x), int(y)), 4, (0, 0, 200), -1)

    # Green = survived filtering
    px, py = project_points(filtered_points, P, h, w)
    for x, y in zip(px, py):
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)

    scale = 800 / max(h, w)
    display = cv2.resize(out, (int(w * scale), int(h * scale)))
    out_path = os.path.join(config.OUTPUT_DIR, f"projection_frame{frame_idx:02d}.jpg")
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path} — green={len(px)} kept, red=rejected")
    try:
        cv2.imshow(f"Frame {frame_idx}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("No display — check the saved file")

if __name__ == "__main__":
    main()