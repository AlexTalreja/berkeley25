#!/usr/bin/env python3
"""
screen_stream_overlay_async.py  (v6.2 – GPT-first, regex-fallback, index logging)

* Real-time screen capture + auto-redact.
* Tries GPT (via Letta) first; falls back to regex if GPT errors / times out.
* Prints the list of sensitive indexes chosen for every OCR pass.
"""

from __future__ import annotations
import argparse, platform, subprocess, sys, time, re, json, os, hashlib, concurrent.futures
from typing import List, Tuple, Optional

import cv2, numpy as np
from mss import mss
import pytesseract
import mediapipe as mp
from dotenv import load_dotenv
from letta_client import Letta          # GPT via Letta

# ────────────────────────────────────────────────────────────────
# 0. configuration
# ────────────────────────────────────────────────────────────────
load_dotenv()
LETTA_TOKEN   = os.getenv("LETTA_TOKEN", "")
GPT_AGENT_ID  = "agent-43235d24-2143-4166-8b0d-2a9c26b4434a"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ────────────────────────────────────────────────────────────────
# 1. CLI
# ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Screen capture + auto-redact")
parser.add_argument("-w","--window", help="Exact window title (Windows only)")
parser.add_argument("--rtmp", help="RTMP URL (omit ⇒ local preview)")
parser.add_argument("--fps", type=int, default=30, help="Capture frame-rate")
parser.add_argument("--interval", type=int, default=2,
                    help="OCR every N frames")
parser.add_argument("--hold", type=int, default=30,
                    help="Frames to KEEP a box after last seen")
parser.add_argument("--pad", type=int, default=4,
                    help="Pixel padding around every box")
parser.add_argument("--face-blur", type=int, default=75,
                    help="Gaussian kernel size for faces")
parser.add_argument("--face-interval", type=int, default=2,
                    help="Run face detection every N frames")
parser.add_argument("--color", default="0,0,255", help="Debug square BGR")
parser.add_argument("--scale", type=float, default=0.75,
                    help="Down-scale factor for OCR")
parser.add_argument("--gpt-timeout", type=int, default=30,
                    help="Seconds to wait for GPT before regex fallback")
args = parser.parse_args()

GPT_TIMEOUT   = args.gpt_timeout

box_color   = tuple(map(int, args.color.split(",")))
frame_delay = 1.0 / args.fps
face_k      = max(3, args.face_blur | 1)
text_k      = 35

# ────────────────────────────────────────────────────────────────
# 2. detectors
# ────────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.4)

API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}")
PHRASE_RE  = re.compile(
    r"(password|passwd|secret|apikey|token|jwt|bearer|session|pin|cvv|ssn|dob"
    r"|address|phone|email)", re.I)

client = Letta(token=LETTA_TOKEN) if LETTA_TOKEN else None
bg_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # OCR + GPT
ocr_future: Optional[concurrent.futures.Future] = None
last_text_hash = ""

# ────────────────────────────────────────────────────────────────
# helper classes / functions
# ────────────────────────────────────────────────────────────────
def expand_box(b, pad, W, H):
    x1,y1,x2,y2 = b
    return (max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad))

def blur_region(img, x1, y1, x2, y2, k):
    sub = img[y1:y2, x1:x2]
    if sub.size:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(sub, (k,k), 0)

def iou(b1, b2):
    x1 = max(b1[0],b2[0]); y1 = max(b1[1],b2[1])
    x2 = min(b1[2],b2[2]); y2 = min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    if not inter: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(a1 + a2 - inter)

class PersistentBoxes:
    def __init__(self, hold, thresh=0.5):
        self.hold, self.thresh, self.items = hold, thresh, []
    def update(self, new_boxes: List[Tuple[int,int,int,int]]):
        for it in self.items: it["ttl"] -= 1
        for nb in new_boxes:
            for it in self.items:
                if iou(nb, it["box"]) >= self.thresh:
                    it["box"], it["ttl"] = nb, self.hold
                    break
            else:
                self.items.append({"box": nb, "ttl": self.hold})
        self.items = [it for it in self.items if it["ttl"] > 0]
    def boxes(self): return [it["box"] for it in self.items]

def pick_monitor(sct, title):
    if title and platform.system()=="Windows":
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            sys.exit(f'Window \"{title}\" not found.')
        l,t,r,b = win32gui.GetWindowRect(hwnd)
        return {"left":l,"top":t,"width":r-l,"height":b-t}
    if title: sys.exit("Window capture by title is Windows-only.")
    return sct.monitors[0]

# ────────────────────────────────────────────────────────────────
# 3. OCR + GPT (background thread) — GPT first, regex fallback
# ────────────────────────────────────────────────────────────────
def gpt_sensitive_indexes(prompt: str) -> Optional[List[int]]:
    """Call Letta; return list, or None on failure/timeout."""
    if not client:
        return None
    future = bg_pool.submit(
        client.agents.messages.create,
        agent_id=GPT_AGENT_ID,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        result = future.result(timeout=GPT_TIMEOUT)
        for m in result.messages:
            if m.message_type == "assistant_message":
                return json.loads(m.content)
    except Exception as e:
        print(f"[redact] GPT timeout/error ({GPT_TIMEOUT}s) → regex fallback")
    return None

def ocr_and_classify(gray_small: np.ndarray, scale: float
    ) -> List[Tuple[int,int,int,int]]:
    """OCR frame → GPT (or regex) → bounding boxes."""
    data = pytesseract.image_to_data(
        gray_small,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6"
    )

    texts, idx_map = [], []
    for i, txt in enumerate(data["text"]):
        if txt:
            idx_map.append(i)
            texts.append(txt)

    if not texts:
        return []

    prompt = (
        "You are a security assistant looking at raw OCR output.\n"
        "Return indexes of any string that might be sensitive. "
        "Respond ONLY with a JSON array of integers.\n\n"
        "### Strings (index : text)\n" +
        "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    )
    sensitive_idx = gpt_sensitive_indexes(prompt)

    if sensitive_idx is None:  # regex fallback
        sensitive_idx = [
            i for i, t in enumerate(texts)
            if API_KEY_RE.search(t) or PHRASE_RE.search(t)
        ]

    # ---- LOG what we found ------------------------------------
    print(f"[redact] indexes={sensitive_idx}  source={'GPT' if client and sensitive_idx else 'regex'}")

    inv = 1.0 / scale
    boxes = []
    for idx in sensitive_idx:
        if idx >= len(idx_map): continue
        i = idx_map[idx]
        x,y,w,h = (data[k][i] for k in ("left","top","width","height"))
        boxes.append((int(x*inv), int(y*inv), int((x+w)*inv), int((y+h)*inv)))
    return boxes

# ────────────────────────────────────────────────────────────────
# 4. main loop
# ────────────────────────────────────────────────────────────────
with mss() as sct:
    mon = pick_monitor(sct, args.window)
    W, H = mon["width"], mon["height"]

    face_cache = PersistentBoxes(args.hold)
    text_cache = PersistentBoxes(args.hold)

    ffmpeg = None
    if args.rtmp:
        ffmpeg = subprocess.Popen([
            "ffmpeg","-loglevel","error","-y",
            "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}",
            "-r",str(args.fps),"-i","-","-c:v","libx264",
            "-preset","veryfast","-pix_fmt","yuv420p",
            "-f","flv",args.rtmp], stdin=subprocess.PIPE)

    frame_i = 0
    try:
        while True:
            t0 = time.time()
            raw = np.ascontiguousarray(np.array(sct.grab(mon))[:,:,:3])

            # Faces
            if frame_i % args.face_interval == 0:
                faces = []
                res = mp_face.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                if res.detections:
                    for d in res.detections:
                        bb = d.location_data.relative_bounding_box
                        faces.append((int(bb.xmin*W), int(bb.ymin*H),
                                      int((bb.xmin+bb.width)*W),
                                      int((bb.ymin+bb.height)*H)))
                faces = [expand_box(b,args.pad,W,H) for b in faces]
                face_cache.update(faces)

            # OCR / GPT async
            if frame_i % args.interval == 0:
                small = cv2.resize(raw, None, fx=args.scale, fy=args.scale,
                                   interpolation=cv2.INTER_AREA)
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                img_hash = hashlib.md5(gray_small.tobytes()).hexdigest()
                if img_hash != last_text_hash:
                    last_text_hash = img_hash
                    if ocr_future is None or ocr_future.done():
                        ocr_future = bg_pool.submit(
                            ocr_and_classify, gray_small, args.scale)

            # async done?
            if ocr_future and ocr_future.done():
                boxes = ocr_future.result()
                boxes = [expand_box(b,args.pad,W,H) for b in boxes]
                text_cache.update(boxes)
                ocr_future = None

            # Blur & display / stream
            img = raw.copy()
            for x1,y1,x2,y2 in face_cache.boxes():
                blur_region(img,x1,y1,x2,y2,face_k)
            for x1,y1,x2,y2 in text_cache.boxes():
                blur_region(img,x1,y1,x2,y2,text_k)

            if ffmpeg:
                ffmpeg.stdin.write(img.tobytes())
            else:
                cv2.imshow("Preview – press Q to quit", img)
                if cv2.waitKey(1) & 0xFF in (ord('q'),ord('Q')):
                    break

            # keep target FPS
            dt = time.time() - t0
            if dt < frame_delay:
                time.sleep(frame_delay - dt)
            frame_i += 1

    except KeyboardInterrupt:
        pass
    finally:
        bg_pool.shutdown(wait=False)
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        cv2.destroyAllWindows()
