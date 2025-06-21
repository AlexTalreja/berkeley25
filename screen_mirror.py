#!/usr/bin/env python3
"""
screen_stream_overlay.py
Capture full screen (or one window) → draw a solid rectangle in the
top-left corner → preview locally *or* pipe to an RTMP server.

$ python screen_stream_overlay.py                 # preview desktop
$ python screen_stream_overlay.py -w "Zoom"       # capture "Zoom" window
$ python screen_stream_overlay.py --rtmp rtmp://live.twitch.tv/app/<key>
"""
import argparse
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
from mss import mss


# --------------------------------------------------------------------------- #
# 1. command-line arguments
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Screen capture + overlay")
parser.add_argument("-w", "--window",
                    help="Exact window title to capture (Windows only). "
                         "If omitted, captures the full desktop.")
parser.add_argument("--rtmp",
                    help="RTMP URL to stream to (omit => local preview)")
parser.add_argument("--fps", type=int, default=30, help="Capture frame-rate")
parser.add_argument("--box", type=int, default=50,
                    help="Size (px) of the top-left overlay square")
parser.add_argument("--color", default="255,0,0",
                    help="Box BGR color, comma-separated (default red)")
args = parser.parse_args()
box_color = tuple(map(int, args.color.split(",")))   # e.g. "0,255,0"

frame_delay = 1.0 / args.fps


# --------------------------------------------------------------------------- #
# 2. choose capture region
# --------------------------------------------------------------------------- #
def pick_monitor(sct: mss, title: str | None):
    if title and platform.system() == "Windows":
        # naive title match on Windows
        try:
            import win32gui
        except ImportError:
            sys.exit("`pywin32` is required for window capture on Windows:\n"
                     "    pip install pywin32")
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            sys.exit(f'Window titled "{title}" not found.')
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        return {"left": l, "top": t, "width": r - l, "height": b - t}
    elif title:
        sys.exit("Window capture by title is only implemented on Windows.")
    else:
        # monitor #0 is the virtual full desktop for mss
        return sct.monitors[0]


# --------------------------------------------------------------------------- #
# 3. main
# --------------------------------------------------------------------------- #
with mss() as sct:
    monitor = pick_monitor(sct, args.window)
    w, h = monitor["width"], monitor["height"]

    # optional ffmpeg pipe ---------------------------------------------------- #
    ffmpeg = None
    if args.rtmp:
        ff_cmd = [
            "ffmpeg",
            "-loglevel", "error",      # keep console clean
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(args.fps),
            "-i", "-",                 # read frames from stdin
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-f", "flv",
            args.rtmp,
        ]
        ffmpeg = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    try:
        while True:
            t0 = time.time()

            # grab → numpy array (BGRA) → drop alpha → make contiguous
            frame = sct.grab(monitor)
            img = np.ascontiguousarray(np.array(frame)[:, :, :3])

            # overlay rectangle
            cv2.rectangle(img, (0, 0), (args.box, args.box),
                          box_color, thickness=-1)

            # output
            if ffmpeg:
                ffmpeg.stdin.write(img.tobytes())
            else:
                cv2.imshow("Live Preview – press Q to quit", img)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break

            # crude FPS limiter
            dt = time.time() - t0
            if dt < frame_delay:
                time.sleep(frame_delay - dt)

    except KeyboardInterrupt:
        pass
    finally:
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        cv2.destroyAllWindows()
