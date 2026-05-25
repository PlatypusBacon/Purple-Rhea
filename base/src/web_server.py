"""
Web server for Purple-Rhea base node.
Serves a dashboard UI and exposes an API for scan control.
Run on the Pi; access from any device on the same Wi-Fi network.
"""

import json
import os
import queue
import threading
import traceback

from flask import Flask, Response, jsonify, request, send_from_directory, send_file

import config

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "output")
FRAMES_DIR = os.path.join(OUTPUT_ROOT, "frames")
MESH_DIR = os.path.join(OUTPUT_ROOT, "mesh")
POINT_DIR = os.path.join(OUTPUT_ROOT, "point")

app = Flask(__name__, static_folder="web/static", static_url_path="/static")

# ── Global scan state ────────────────────────────────────────────────────────

class ScanState:
    IDLE = "idle"
    SCANNING = "scanning"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"

scan_lock = threading.Lock()
scan_status = {
    "state": ScanState.IDLE,
    "frames_captured": 0,
    "total_frames": config.TOTAL_FRAMES,
    "pipeline_stage": "",
    "error": "",
}
event_subscribers: list[queue.Queue] = []


def publish_event(event_type: str, data: dict | None = None):
    payload = {"type": event_type, **scan_status}
    if data:
        payload.update(data)
    msg = f"data: {json.dumps(payload)}\n\n"
    dead = []
    for q in event_subscribers:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    for q in dead:
        event_subscribers.remove(q)


# ── Scan thread ──────────────────────────────────────────────────────────────

def on_scan_triggered():
    """Called when the Start Scan button is pressed."""
    from main import main
    main()


def _scan_thread():
    try:
        scan_status["state"] = ScanState.SCANNING
        scan_status["frames_captured"] = 0
        scan_status["pipeline_stage"] = ""
        scan_status["error"] = ""
        publish_event("scan_start")

        on_scan_triggered()

        scan_status["state"] = ScanState.DONE
        publish_event("scan_done")

    except Exception as e:
        scan_status["state"] = ScanState.ERROR
        scan_status["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        publish_event("scan_error")


# ── API routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify(scan_status)


@app.route("/api/scan/start", methods=["POST"])
def api_scan_start():
    with scan_lock:
        if scan_status["state"] in (ScanState.SCANNING, ScanState.PROCESSING):
            return jsonify({"error": "Scan already in progress"}), 409

    t = threading.Thread(target=_scan_thread, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/events")
def api_events():
    """Server-Sent Events stream for live progress updates."""
    q: queue.Queue = queue.Queue(maxsize=50)
    event_subscribers.append(q)

    def stream():
        q.put(f"data: {json.dumps(scan_status)}\n\n")
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            if q in event_subscribers:
                event_subscribers.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/frames")
def api_frames():
    """List captured frame images."""
    frame_dir = FRAMES_DIR
    if not os.path.isdir(frame_dir):
        return jsonify([])
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    return jsonify(files)


@app.route("/api/frames/<filename>")
def api_frame_image(filename):
    return send_from_directory(FRAMES_DIR, filename)


@app.route("/api/result/download")
def api_result_download():
    """Download the reconstructed .obj file."""
    if not os.path.isdir(MESH_DIR):
        return jsonify({"error": "No result available"}), 404
    objs = [f for f in os.listdir(MESH_DIR) if f.endswith(".obj")]
    if not objs:
        return jsonify({"error": "No result available"}), 404
    return send_file(os.path.join(MESH_DIR, objs[0]), as_attachment=True)


@app.route("/api/result/points")
def api_result_points():
    """Return point cloud as JSON for the 3D viewer."""
    if not os.path.isdir(POINT_DIR):
        return jsonify({"error": "No point cloud available"}), 404

    for name in sorted(os.listdir(POINT_DIR)):
        path = os.path.join(POINT_DIR, name)
        if name.endswith(".npy"):
            import numpy as np
            pts = np.load(path)
            return jsonify(pts.tolist())
        if name.endswith(".obj"):
            points = []
            with open(path) as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if points:
                return jsonify(points)

    return jsonify({"error": "No point cloud available"}), 404


@app.route("/api/result/mesh")
def api_result_mesh():
    """Return mesh vertices and faces as JSON for the 3D viewer."""
    if not os.path.isdir(MESH_DIR):
        return jsonify({"error": "No mesh available"}), 404

    for name in sorted(os.listdir(MESH_DIR)):
        if name.endswith(".obj"):
            verts = []
            faces = []
            with open(os.path.join(MESH_DIR, name)) as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith("f "):
                        parts = line.split()[1:]
                        face = [int(p.split("/")[0]) - 1 for p in parts]
                        faces.append(face)
            if verts:
                return jsonify({"vertices": verts, "faces": faces})

    return jsonify({"error": "No mesh available"}), 404


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(POINT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
