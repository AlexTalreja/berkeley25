#!/usr/bin/env python3
"""
screen_stream_overlay.py  (v5.2 – strong face-blur)

* Captures the desktop or a single window.
* Detects human faces & sensitive text.
* Blurs:
    • Faces  with a large kernel  (--face-blur, default 75)
    • Text   with a modest kernel (35)
* Holds every blur for --hold frames after last detection (anti-flicker).
"""

import argparse, platform, subprocess, sys, time, re
from pathlib import Path

import cv2, numpy as np
from mss import mss
import pytesseract
import mediapipe as mp
import openai
import json
import os
from dotenv import load_dotenv
from google import genai

# Path to the Tesseract executable (adjust if yours lives elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
client = genai.Client(api_key="")

load_dotenv() 
# ────────────────────────────────────────────────────────────────
# 1. command-line arguments
# ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Screen capture + auto-redact")
parser.add_argument("-w", "--window", help="Exact window title (Windows only)")
parser.add_argument("--rtmp", help="RTMP URL (omit => local preview)")
parser.add_argument("--fps", type=int, default=30, help="Capture frame-rate")
parser.add_argument("--interval", type=int, default=1,
                    help="OCR every N frames (keep at 1 for zero-leak)")
parser.add_argument("--hold", type=int, default=15,
                    help="Frames to KEEP a box after last seen (anti-flicker)")
parser.add_argument("--pad", type=int, default=4,
                    help="Extra pixel padding around every box")
parser.add_argument("--face-blur", type=int, default=75,
                    help="Gaussian-kernel size for faces (odd; higher = stronger)")
parser.add_argument("--box", type=int, default=50, help="Demo overlay square")
parser.add_argument("--color", default="255,0,0", help="Overlay BGR color")
args = parser.parse_args()

box_color   = tuple(map(int, args.color.split(",")))
frame_delay = 1.0 / args.fps
face_k      = 100       
text_k      = 35                               # legacy value

# ────────────────────────────────────────────────────────────────
# 2. detectors
# ────────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.4)

API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}")
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed
)

# OPEN AI STUFF

openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4o-mini"          # fast & cheap enough for frame-level use
GPT_FUNC = {
    "name": "mark_sensitive",
    "description": "Return indexes of sensitive strings.",
    "parameters": {
        "type": "object",
        "properties": {
            "indexes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "0-based indexes that MUST be redacted."
            }
        },
        "required": ["indexes"],
    },
}
def find_sensitive_text(gray_img) -> list[tuple[int,int,int,int]]:
    """
    1. OCR the frame exactly as before (pytesseract).
    2. Ask GPT which snippets are sensitive.
    3. Return their bounding boxes.
    """
    data = pytesseract.image_to_data(
        gray_img, output_type=pytesseract.Output.DICT
    )

    # Collect candidate strings ≥ 4 chars to keep the prompt short
    texts, idx_map = [], []
    for i, txt in enumerate(data["text"]):
        if txt and len(txt) >= 4:
            idx_map.append(i)     # map from compact index -> tesseract index
            texts.append(txt)

    if not texts:
        return []

    # --------  Call GPT  ----------------------------------------------------
    prompt = (
        "You are a security assistant looking at raw OCR output.\n"
        "For any string that is *definitely* sensitive (API keys, "
        "passwords, tokens, credit-card numbers, SSNs, private addresses, "
        "personal phone numbers, etc.) you must return its index. For testing purposes now, treat the literal word password as sensitive\n\n"
        "### Strings (index : text)\n" +
        "\n".join(f"{i}: {t}" for i, t in enumerate(texts)) + "\n\n"
        "Respond *only* with JSON that matches the function schema."
    )
    try:
        # rsp = openai.chat.completions.create(
        #     model=GPT_MODEL,
        #     messages=[{"role": "system", "content": "You are a helpful assistant."},
        #               {"role": "user", "content": prompt}],
        #     functions=[GPT_FUNC],
        #     function_call={"name": "mark_sensitive"},
        #     temperature=0.0,
        # )
        # sensitive = json.loads(rsp.choices[0].message.function_call.arguments)["indexes"]
        # print(sensitive)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        sensitive = json.loads(response.text)
        print(sensitive)
    except Exception as e:
        print("OpenAI failure, falling back to old regex:", e)
        sensitive = []   # (or call your old regex here as a backup)

    # --------  Convert indexes → bounding boxes  ----------------------------
    boxes = []
    for compact_idx in sensitive:
        i = idx_map[compact_idx]
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        boxes.append((x, y, x + w, y + h))
    return boxes


# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def expand_box(b, pad, W, H):
    x1, y1, x2, y2 = b
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(W, x2 + pad), min(H, y2 + pad))

def blur_region(img, x1, y1, x2, y2, k):
    sub = img[y1:y2, x1:x2]
    if sub.size:
        k = max(3, k | 1)  # odd, ≥3
        img[y1:y2, x1:x2] = cv2.GaussianBlur(sub, (k, k), 0)

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if not inter: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(a1 + a2 - inter)

class PersistentBoxes:
    def __init__(self, hold, thresh=0.5):
        self.hold = hold; self.thresh = thresh; self.items = []
    def update(self, new_boxes):
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
    if title and platform.system() == "Windows":
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd: sys.exit(f'Window "{title}" not found.')
        l,t,r,b = win32gui.GetWindowRect(hwnd)
        return {"left":l,"top":t,"width":r-l,"height":b-t}
    if title: sys.exit("Window capture by title is Windows-only.")
    return sct.monitors[0]

# ────────────────────────────────────────────────────────────────
# 3. main
# ────────────────────────────────────────────────────────────────
with mss() as sct:
    monitor = pick_monitor(sct, args.window)
    W, H = monitor["width"], monitor["height"]

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
            raw = np.ascontiguousarray(np.array(sct.grab(monitor))[:, :, :3])

            # ── Detect faces every frame ────────────────────────
            faces = []
            res = mp_face.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            if res.detections:
                for d in res.detections:
                    bb = d.location_data.relative_bounding_box
                    faces.append((int(bb.xmin*W), int(bb.ymin*H),
                                  int((bb.xmin+bb.width)*W),
                                  int((bb.ymin+bb.height)*H)))
            faces = [expand_box(b, args.pad, W, H) for b in faces]
            face_cache.update(faces)

            # ── OCR every --interval frames ─────────────────────
            if frame_i % args.interval == 0:
                txt = find_sensitive_text(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY))
                txt = [expand_box(b, args.pad, W, H) for b in txt]
                text_cache.update(txt)

            # ── Blur & present ─────────────────────────────────
            img = raw.copy()
            for x1,y1,x2,y2 in face_cache.boxes():
                blur_region(img, x1,y1,x2,y2, face_k)   # strong
            for x1,y1,x2,y2 in text_cache.boxes():
                blur_region(img, x1,y1,x2,y2, text_k)   # moderate

            cv2.rectangle(img,(0,0),(args.box,args.box),box_color,-1)

            if ffmpeg: ffmpeg.stdin.write(img.tobytes())
            else:
                cv2.imshow("Preview – press Q to quit", img)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break

            dt = time.time()-t0
            if dt < frame_delay: time.sleep(frame_delay-dt)
            frame_i += 1

    except KeyboardInterrupt:
        pass
    finally:
        if ffmpeg: ffmpeg.stdin.close(); ffmpeg.wait()
        cv2.destroyAllWindows()
