#!/usr/bin/env python3
"""
screen_stream_overlay_async.py  (v7.5 – Gemini-preferential blur boxes)

• Captures the desktop (or a chosen window), detects faces & sensitive text,
  and live-blurs them.
• Gemini is queried on every OCR batch; regex is only a backup.
• Works with the current google-generativeai package (no request_options).
• Console logs each blur hit and Gemini’s raw / parsed outputs.
"""

from __future__ import annotations
import argparse, platform, subprocess, sys, time, re, os, json, ast
from concurrent.futures import ThreadPoolExecutor, Future

import cv2, numpy as np
from mss import mss
import pytesseract, mediapipe as mp
from dotenv import load_dotenv
import google.generativeai as genai          # ← modern import path for Gemini

# ── OCR / regex basics ──────────────────────────────────────────────────────
TESS_PATH   = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESS_PATH
TESS_CONFIG = "--oem 3 --psm 11 -l eng"

API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}", re.I)
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)

# ── Gemini setup ────────────────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-1.5-flash-latest"   # use a fast, low-cost text model
model      = genai.GenerativeModel(MODEL_NAME)

# ── Gemini prompt template ──────────────────────────────────────────────────
PROMPT = """You are a security assistant. Here are OCR lines (index : text):

{strings}

Return ONLY a JSON array of the indexes (ints) of lines that should be blurred
because they MIGHT reveal sensitive information.

Sensitive examples:
• Passwords, passphrases, API tokens, secrets
• Phone numbers (any format)
• Street or mailing addresses
• Credit-card or bank numbers
• E-mail addresses
• Personal names when combined with other data

If no line is sensitive, return [].

### Example
Input
0: foo
1: sk_live_ABC…
→ [1]
"""

# ── CLI ---------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Screen capture + auto-redact (v7.5)")
ap.add_argument("-w", "--window", help="Exact window title (Windows only)")
ap.add_argument("--rtmp", help="RTMP URL (omit ⇒ preview locally)")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--interval", type=int, default=1)
ap.add_argument("--hold", type=int, default=40)
ap.add_argument("--pad", type=int, default=4)
ap.add_argument("--scale", type=float, default=1.5)
ap.add_argument("--face-blur", type=int, default=75)
ap.add_argument("--color", default="255,0,0")
args = ap.parse_args()

box_color   = tuple(int(c) for c in args.color.split(","))
frame_delay = 1 / args.fps
face_k      = max(3, args.face_blur | 1)
text_k      = 35

# ── detectors ----------------------------------------------------------------
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6)

# ── helpers ------------------------------------------------------------------
def preprocess(gray: np.ndarray) -> np.ndarray:
    sharp = cv2.addWeighted(gray, 1.5,
                            cv2.GaussianBlur(gray, (0, 0), 1.2),
                            -0.5, 0)
    bw = cv2.adaptiveThreshold(sharp, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 35, 11)
    return cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)

_json_re = re.compile(r"\[[^\]]*\]")

def robust_json_extract(text: str) -> list[int]:
    for parser in (ast.literal_eval, json.loads):
        try:
            out = parser(text.strip())
            if isinstance(out, list):
                return [int(x) for x in out]
        except Exception:
            pass
    m = _json_re.search(text)
    if m:
        try:
            return [int(x) for x in json.loads(m.group())]
        except Exception:
            pass
    return []

def llm_sensitive(lines: list[str]) -> list[int]:
    prompt = PROMPT.format(strings="\n".join(f"{i}: {t}"
                                             for i, t in enumerate(lines)))
    try:
        rsp  = model.generate_content(prompt, stream=False)
        text = (rsp.text if hasattr(rsp, "text")
                        else rsp.candidates[0].content.parts[0].text)
        #print("[Gemini raw]", text)
        idxs = robust_json_extract(text)
        print("[Gemini parsed]", idxs)
        return idxs
    except Exception as e:
        print("[Gemini error]", e)
        return []

# ── OCR → blur-box extractor -------------------------------------------------
def find_sensitive(gray: np.ndarray, scale: float):
    if scale != 1.0:
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    d = pytesseract.image_to_data(
        preprocess(gray),
        output_type=pytesseract.Output.DICT,
        config=TESS_CONFIG)

    # Group words → lines
    lines, boxes, key, words, coords = [], [], None, [], []
    for i, txt in enumerate(d["text"]):
        if not txt:
            continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key and words:
            xs, ys, x2s, y2s = zip(*coords)
            lines.append(" ".join(words))
            boxes.append((min(xs), min(ys), max(x2s), max(y2s)))
            words, coords = [], []
        key = k
        words.append(txt)
        coords.append((d["left"][i], d["top"][i],
                       d["left"][i] + d["width"][i],
                       d["top"][i] + d["height"][i]))
    if words:
        xs, ys, x2s, y2s = zip(*coords)
        lines.append(" ".join(words))
        boxes.append((min(xs), min(ys), max(x2s), max(y2s)))

    if not lines:
        return []

    idxs_llm = set(llm_sensitive(lines))
    idxs_rx  = {i for i, t in enumerate(lines)
                if i not in idxs_llm and
                   (API_KEY_RE.search(t) or PHRASE_RE.search(t))}
    all_idxs = idxs_llm | idxs_rx

    for i in sorted(all_idxs):
        method = "LLM" if i in idxs_llm else "regex"
        print(f"[blur] idx={i:<3} via={method:<5} text={lines[i]!r}", flush=True)

    out = []
    for i in all_idxs:
        x1, y1, x2, y2 = boxes[i]
        if scale != 1.0:
            x1, y1, x2, y2 = [int(v / scale) for v in (x1, y1, x2, y2)]
        out.append((x1, y1, x2, y2))
    return out

# ── misc geometry & smoothing helpers ---------------------------------------
def expand(box, pad, W, H):
    x1, y1, x2, y2 = box
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(W, x2 + pad), min(H, y2 + pad))

def blur(img, x1, y1, x2, y2, k):
    roi = img[y1:y2, x1:x2]
    if roi.size:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    if not inter:
        return 0.0
    return inter / float(
        (a[2] - a[0]) * (a[3] - a[1]) +
        (b[2] - b[0]) * (b[3] - b[1]) - inter)

class SmoothPersistent:
    """Keeps blur boxes alive & smooths their edges."""
    def __init__(self, hold, thresh=0.3, smooth=0.3):
        self.hold = hold; self.thresh = thresh; self.smooth = smooth
        self.items = []

    def _ema(self, old, new):
        return int(old * (1 - self.smooth) + new * self.smooth)

    def update(self, new_boxes):
        for it in self.items:
            it["ttl"] -= 1
        for nb in new_boxes:
            for it in self.items:
                if iou(nb, it["box"]) >= self.thresh:
                    ox1, oy1, ox2, oy2 = it["box"]
                    nx1, ny1, nx2, ny2 = nb
                    smoothed = (self._ema(ox1, nx1),
                                self._ema(oy1, ny1),
                                self._ema(ox2, nx2),
                                self._ema(oy2, ny2))
                    if max(abs(smoothed[i] - ox)
                           for i, ox in enumerate(it["box"])) <= 2:
                        smoothed = it["box"]
                    it["box"], it["ttl"] = smoothed, self.hold
                    break
            else:
                self.items.append({"box": nb, "ttl": self.hold})
        self.items = [it for it in self.items if it["ttl"] > 0]

    def boxes(self):
        return [it["box"] for it in self.items]

# ── monitor picker -----------------------------------------------------------
def pick_monitor(sct, title):
    if title and platform.system() == "Windows":
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            sys.exit(f'Window "{title}" not found.')
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        return {"left": l, "top": t, "width": r - l, "height": b - t}
    if title:
        sys.exit("Window capture by title is Windows-only.")
    return sct.monitors[0]

# ── main loop ----------------------------------------------------------------
with mss() as sct, ThreadPoolExecutor(max_workers=1) as pool:
    mon = pick_monitor(sct, args.window)
    W, H = mon["width"], mon["height"]

    face_cache = SmoothPersistent(args.hold)
    text_cache = SmoothPersistent(args.hold)

    future: Future | None = None
    last_txt_boxes: list[tuple[int, int, int, int]] = []

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
    try:
        while True:
            start = time.time()
            frame = np.ascontiguousarray(
                np.array(sct.grab(mon))[:, :, :3])

            # face detection ------------------------------------------------
            faces = []
            res = mp_face.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.detections:
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    faces.append((int(bb.xmin * W),
                                  int(bb.ymin * H),
                                  int((bb.xmin + bb.width) * W),
                                  int((bb.ymin + bb.height) * H)))
            face_cache.update([expand(b, args.pad, W, H) for b in faces])

            # OCR / LLM async ----------------------------------------------
            if frame_i % args.interval == 0:
                if future and future.done():
                    try:
                        last_txt_boxes = future.result()
                    except Exception:
                        last_txt_boxes = []
                    text_cache.update(
                        [expand(b, args.pad, W, H) for b in last_txt_boxes])
                    future = None
                if future is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    future = pool.submit(find_sensitive, gray, args.scale)

            # draw / blur ---------------------------------------------------
            for x1, y1, x2, y2 in face_cache.boxes():
                blur(frame, x1, y1, x2, y2, face_k)
            for x1, y1, x2, y2 in text_cache.boxes():
                blur(frame, x1, y1, x2, y2, text_k)

            cv2.rectangle(frame, (0, 0), (40, 40), box_color, -1)

            if ffmpeg:
                ffmpeg.stdin.write(frame.tobytes())
            else:
                cv2.imshow("Preview – Q quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break

            dt = time.time() - start
            if dt < frame_delay:
                time.sleep(frame_delay - dt)
            frame_i += 1

    finally:
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        cv2.destroyAllWindows()
