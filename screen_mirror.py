#!/usr/bin/env python3
"""
screen_stream_overlay.py  (v2 – with automatic redaction)

* Captures the desktop or a single window.
* Detects human faces + sensitive text (API keys, “password”, etc.).
* Blurs those regions in-line, then previews locally or streams to RTMP.
"""

import argparse, platform, subprocess, sys, time, re
from pathlib import Path

import cv2, numpy as np
from mss import mss
import pytesseract
import mediapipe as mp

# ------------------------------------------------------------------ #
# 1. command-line arguments
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Screen capture + auto-redact")
parser.add_argument("-w", "--window", help="Exact window title (Windows only)")
parser.add_argument("--rtmp", help="RTMP URL (omit => local preview)")
parser.add_argument("--fps", type=int, default=30, help="Capture frame-rate")
parser.add_argument("--interval", type=int, default=3,
                    help="Run text OCR every N frames (perf tweak)")
parser.add_argument("--box", type=int, default=50, help="Demo overlay square")
parser.add_argument("--color", default="255,0,0", help="Overlay BGR color")
args = parser.parse_args()
box_color = tuple(map(int, args.color.split(",")))
frame_delay = 1.0 / args.fps

# ------------------------------------------------------------------ #
# 2. detectors
# ------------------------------------------------------------------ #
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.4)

API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}")
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)

def find_sensitive_text(gray_img) -> list[tuple[int, int, int, int]]:
    """Return [x1,y1,x2,y2] boxes whose text matches the regexes."""
    boxes = []
    data = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT)
    for i, txt in enumerate(data["text"]):
        if not txt or len(txt) < 4:
            continue
        if API_KEY_RE.search(txt) or PHRASE_RE.search(txt):
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            boxes.append((x, y, x + w, y + h))
    return boxes

def blur_region(img, x1, y1, x2, y2, k=35):
    sub = img[y1:y2, x1:x2]
    if sub.size:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(sub, (k, k), 0)

# ------------------------------------------------------------------ #
# 3. pick capture region
# ------------------------------------------------------------------ #
def pick_monitor(sct: mss, title: str | None):
    if title and platform.system() == "Windows":
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            sys.exit(f'Window "{title}" not found.')
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        return {"left": l, "top": t, "width": r - l, "height": b - t}
    elif title:
        sys.exit("Window capture by title is Windows-only.")
    return sct.monitors[0]

# ------------------------------------------------------------------ #
# 4. main loop
# ------------------------------------------------------------------ #
with mss() as sct:
    monitor = pick_monitor(sct, args.window)
    W, H = monitor["width"], monitor["height"]

    # optional ffmpeg pipe
    ffmpeg = None
    if args.rtmp:
        ffmpeg = subprocess.Popen([
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
            "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
            "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-f", "flv", args.rtmp
        ], stdin=subprocess.PIPE)

    frame_i = 0
    cached_text_boxes, cached_face_boxes = [], []

    try:
        while True:
            t0 = time.time()

            # capture
            frame = sct.grab(monitor)
            img = np.ascontiguousarray(np.array(frame)[:, :, :3])

            # -------------------------------------------------- #
            # 4A. face detection every frame
            results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cached_face_boxes = []
            if results.detections:
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = int(bb.xmin * W); y1 = int(bb.ymin * H)
                    x2 = int((bb.xmin + bb.width) * W)
                    y2 = int((bb.ymin + bb.height) * H)
                    cached_face_boxes.append((x1, y1, x2, y2))

            # 4B. OCR every N frames (tweak with --interval)
            if frame_i % args.interval == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cached_text_boxes = find_sensitive_text(gray)

            # 4C. apply blur
            for x1, y1, x2, y2 in cached_face_boxes + cached_text_boxes:
                blur_region(img, x1, y1, x2, y2)

            # demo overlay square (unchanged from original)
            cv2.rectangle(img, (0, 0), (args.box, args.box), box_color, -1)

            # output
            if ffmpeg:
                ffmpeg.stdin.write(img.tobytes())
            else:
                cv2.imshow("Preview – press Q to quit", img)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break

            # FPS limiter
            dt = time.time() - t0
            if dt < frame_delay:
                time.sleep(frame_delay - dt)
            frame_i += 1

    except KeyboardInterrupt:
        pass
    finally:
        if ffmpeg:
            ffmpeg.stdin.close(); ffmpeg.wait()
        cv2.destroyAllWindows()
