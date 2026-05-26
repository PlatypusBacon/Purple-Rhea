#!/usr/bin/env python3
"""
Live plot of LOCATION:<roll>,<pitch>,<yaw> lines from the XIAO BLE Sense
tracker over USB CDC-ACM.

Usage:
    python3 plot_live.py /dev/cu.usbmodem1101
    python3 plot_live.py /dev/ttyACM0 --window 30

Quit: close the window, or Ctrl-C in the terminal.
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def reader_thread(port: str, baud: int,
                  t_buf, r_buf, p_buf, y_buf, radius_box,
                  t0: float, stop_evt: threading.Event) -> None:
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"could not open {port}: {e}", file=sys.stderr)
        stop_evt.set()
        return
    ser.dtr = False
    ser.rts = False

    while not stop_evt.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except serial.SerialException:
            break

        if line.startswith("RADIUS:"):
            try:
                radius_box[0] = float(line[len("RADIUS:"):])
            except ValueError:
                pass
            continue

        if not line.startswith("LOCATION:"):
            continue
        try:
            parts = line[len("LOCATION:"):].split(",")
            roll, pitch, yaw = (float(p) for p in parts[:3])
        except (IndexError, ValueError):
            continue
        t_buf.append(time.monotonic() - t0)
        r_buf.append(roll)
        p_buf.append(pitch)
        y_buf.append(yaw)
    ser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="serial port, e.g. /dev/cu.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200,
                    help="ignored over CDC-ACM but required by pyserial")
    ap.add_argument("--window", type=float, default=20.0,
                    help="rolling window in seconds (default 20)")
    ap.add_argument("--maxlen", type=int, default=4000,
                    help="max samples kept (default 4000 ≈ 200 s at 20 Hz)")
    args = ap.parse_args()

    t_buf = collections.deque(maxlen=args.maxlen)
    r_buf = collections.deque(maxlen=args.maxlen)
    p_buf = collections.deque(maxlen=args.maxlen)
    y_buf = collections.deque(maxlen=args.maxlen)

    t0 = time.monotonic()
    stop_evt = threading.Event()
    radius_box = [None]  # single-cell mutable container shared with reader

    th = threading.Thread(
        target=reader_thread,
        args=(args.port, args.baud, t_buf, r_buf, p_buf, y_buf,
              radius_box, t0, stop_evt),
        daemon=True,
    )
    th.start()

    fig, (ax_lines, ax_compass) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.canvas.manager.set_window_title(f"Tracker live — {args.port}")

    (l_roll,)  = ax_lines.plot([], [], label="roll",  color="#d33")
    (l_pitch,) = ax_lines.plot([], [], label="pitch", color="#3a3")
    (l_yaw,)   = ax_lines.plot([], [], label="yaw",   color="#36c")
    ax_lines.set_xlabel("time (s)")
    ax_lines.set_ylabel("angle (deg)")
    ax_lines.set_ylim(-200, 200)
    ax_lines.axhline(0, color="#888", lw=0.5)
    ax_lines.grid(True, alpha=0.3)
    ax_lines.legend(loc="upper left")

    ax_compass.set_aspect("equal")
    ax_compass.set_xlim(-1.2, 1.2)
    ax_compass.set_ylim(-1.2, 1.2)
    ax_compass.set_xticks([])
    ax_compass.set_yticks([])
    ax_compass.set_title("yaw")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#888")
    ax_compass.add_patch(circle)
    for deg, label in [(0, "0"), (90, "90"), (180, "±180"), (-90, "-90")]:
        import math
        rad = math.radians(90 - deg)  # 0° at top, clockwise positive
        ax_compass.text(1.12 * math.cos(rad), 1.12 * math.sin(rad),
                        label, ha="center", va="center", fontsize=9,
                        color="#666")
    (needle,) = ax_compass.plot([0, 0], [0, 1], color="#36c", lw=2.5)
    yaw_text = ax_compass.text(0, -1.15, "", ha="center", va="top",
                               fontsize=10, color="#36c")
    radius_text = ax_compass.text(0, -1.45, "radius: --", ha="center",
                                  va="top", fontsize=10, color="#444")

    def update(_frame):
        if not t_buf:
            return l_roll, l_pitch, l_yaw, needle, yaw_text, radius_text

        t = list(t_buf)
        r = list(r_buf)
        p = list(p_buf)
        y = list(y_buf)

        t_now = t[-1]
        l_roll.set_data(t, r)
        l_pitch.set_data(t, p)
        l_yaw.set_data(t, y)

        ax_lines.set_xlim(max(0.0, t_now - args.window), max(args.window, t_now))

        import math
        yaw_now = y[-1]
        rad = math.radians(90 - yaw_now)
        needle.set_data([0, math.cos(rad)], [0, math.sin(rad)])
        yaw_text.set_text(f"{yaw_now:+.1f}°")

        if radius_box[0] is not None:
            radius_text.set_text(f"radius: {radius_box[0]*100:.1f} cm")
        else:
            radius_text.set_text("radius: (waiting for motion)")

        return l_roll, l_pitch, l_yaw, needle, yaw_text, radius_text

    anim = animation.FuncAnimation(fig, update, interval=50, blit=False,
                                   cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop_evt.set()
        th.join(timeout=1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
