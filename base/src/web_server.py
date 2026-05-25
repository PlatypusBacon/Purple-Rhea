"""
Web server for Purple-Rhea base node.
Serves a dashboard UI and exposes an API for scan control.
Run on the Pi; access from any device on the same Wi-Fi network.
"""

import json
import os
import queue
import threading
import time
import traceback

from flask import Flask, Response, jsonify, request, send_from_directory, send_file

import config
from storage.scan_session import ScanSession

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
    "result_path": "",
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


# ── Pipeline monkeypatch to capture progress ─────────────────────────────────

_original_print = print

def _patched_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    text = " ".join(str(a) for a in args)
    if text.startswith("["):
        scan_status["pipeline_stage"] = text
        publish_event("pipeline_progress")


# ── Scan thread ──────────────────────────────────────────────────────────────

def _scan_thread(source: str):
    import builtins

    try:
        scan_status["state"] = ScanState.SCANNING
        scan_status["frames_captured"] = 0
        scan_status["pipeline_stage"] = ""
        scan_status["error"] = ""
        scan_status["result_path"] = ""
        publish_event("scan_start")

        if source == "serial":
            from comms.serial_receiver import receive_session_serial
            session = _receive_with_progress()
        else:
            from comms.disk_session import start_disk_session
            session = start_disk_session()
            scan_status["frames_captured"] = len(session)
            publish_event("frame_captured")

        session.save_all_jpegs(config.IMAGE_CACHE)

        if len(session) == 0:
            raise RuntimeError("No frames captured")

        scan_status["state"] = ScanState.PROCESSING
        publish_event("processing_start")

        old_print = builtins.print
        builtins.print = _patched_print
        try:
            from pipeline import runner
            obj_path = runner.run(session)
        finally:
            builtins.print = old_print

        scan_status["state"] = ScanState.DONE
        scan_status["result_path"] = obj_path
        publish_event("scan_done")

    except Exception as e:
        scan_status["state"] = ScanState.ERROR
        scan_status["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        publish_event("scan_error")


def _receive_with_progress() -> ScanSession:
    """Wraps serial capture to emit per-frame progress events."""
    import serial
    from comms.tracker_receiver import open_tracker
    from comms.serial_receiver import _wait_for_ready, _receive_one_frame
    from storage.scan_session import ScanFrame, CameraPose

    session = ScanSession()
    ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=5)
    ser.dtr = False
    ser.rts = False

    tracker = open_tracker()
    _wait_for_ready(ser)

    frame_index = 0
    while not session.is_complete():
        ser.write(b"capture\n")
        frame = _receive_one_frame(ser, frame_index, tracker)
        if frame is None:
            frame = _receive_one_frame(ser, frame_index, tracker)
        if frame:
            session.add_frame(frame)
            scan_status["frames_captured"] = len(session)
            publish_event("frame_captured")
        else:
            break
        frame_index += 1

    if tracker:
        tracker.stop()
    ser.close()
    return session


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

    body = request.get_json(silent=True) or {}
    source = body.get("source", "disk")
    t = threading.Thread(target=_scan_thread, args=(source,), daemon=True)
    t.start()
    return jsonify({"status": "started", "source": source})


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
    frame_dir = config.IMAGE_CACHE
    if not os.path.isdir(frame_dir):
        return jsonify([])
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    return jsonify(files)


@app.route("/api/frames/<filename>")
def api_frame_image(filename):
    return send_from_directory(os.path.abspath(config.IMAGE_CACHE), filename)


@app.route("/api/result/download")
def api_result_download():
    """Download the reconstructed .obj file."""
    obj_path = os.path.join(config.OUTPUT_DIR, "reconstruction.obj")
    if not os.path.isfile(obj_path):
        return jsonify({"error": "No result available"}), 404
    return send_file(os.path.abspath(obj_path), as_attachment=True)


@app.route("/api/result/points")
def api_result_points():
    """Return point cloud as JSON for the 3D viewer."""
    import numpy as np
    for name in ("reconstruction_filtered.npy", "reconstruction_raw.npy"):
        path = os.path.join(config.OUTPUT_DIR, name)
        if os.path.isfile(path):
            pts = np.load(path)
            return jsonify(pts.tolist())
    return jsonify({"error": "No point cloud available"}), 404


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify({
        "total_frames": config.TOTAL_FRAMES,
        "step_degrees": config.STEP_DEGREES,
        "nominal_radius": config.NOMINAL_RADIUS,
        "image_width": config.IMAGE_WIDTH,
        "image_height": config.IMAGE_HEIGHT,
        "serial_port": config.SERIAL_PORT,
        "debug_mode": config.DEBUG_MODE,
    })


@app.route("/api/config", methods=["POST"])
def api_config_set():
    body = request.get_json(silent=True) or {}
    allowed = {"total_frames", "step_degrees", "nominal_radius",
               "image_width", "image_height", "serial_port", "debug_mode"}
    updated = {}
    for key, val in body.items():
        if key in allowed:
            setattr(config, key.upper(), val)
            updated[key] = val
    return jsonify({"updated": updated})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_CACHE, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
