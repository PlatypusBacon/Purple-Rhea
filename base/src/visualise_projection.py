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

def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    points_3d  = load_obj(os.path.join(config.OUTPUT_DIR, "reconstruction.obj"))
    projections = np.load(os.path.join(config.OUTPUT_DIR, "projections.npy"))
    print(f"Loaded {len(points_3d)} points, {len(projections)} projections")

    # Load only the one image needed
    session = start_disk_session()
    img = list(session)[frame_idx].load_image()

    P = projections[frame_idx]
    h, w = img.shape[:2]

    # Project
    X_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    proj = (P @ X_h.T).T
    depths = proj[:, 2]
    valid = depths > 0
    px = proj[valid, 0] / depths[valid]
    py = proj[valid, 1] / depths[valid]

    # Draw
    out = img.copy()
    for x, y in zip(px, py):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Scale down for display — full 12MP won't fit on screen
    scale = 800 / max(h, w)
    display = cv2.resize(out, (int(w*scale), int(h*scale)))

    out_path = os.path.join(config.OUTPUT_DIR, f"projection_frame{frame_idx:02d}.jpg")
    cv2.imwrite(out_path, out)
    print(f"Saved {out_path} — {valid.sum()} points projected")

    try:
        cv2.imshow(f"Frame {frame_idx}", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("No display — check the saved file")

if __name__ == "__main__":
    main()