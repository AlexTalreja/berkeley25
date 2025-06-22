#!/usr/bin/env python3
"""
screen_stream_overlay_async.py  (v6.4 – grouped lines + faster path)

* Captures the desktop or a single window.
* Detects faces & sensitive text (API keys, tokens, addresses, etc.).
* Groups OCR tokens into full **lines** so multi-word addresses stay intact.
* Skips the LLM call when a fast regex finds nothing interesting.
* Runs OCR + LLM on a background thread; ~30 fps on mid-range laptops.
"""

from __future__ import annotations
import argparse, platform, subprocess, sys, time, re, os, ast
from concurrent.futures import ThreadPoolExecutor, Future

import cv2, numpy as np
from mss import mss
import pytesseract
import mediapipe as mp
from dotenv import load_dotenv
from google import genai

# ────────────────────────────────────────────────────────────────
# 0. constants & one-time setup
# ────────────────────────────────────────────────────────────────
TESS_PATH   = r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # adjust if needed
pytesseract.pytesseract.tesseract_cmd = TESS_PATH
TESS_CONFIG = "--oem 3 --psm 11 -l eng"          # sparse-text mode

# very fast heuristics to decide if we even need the LLM
QUICK_REGEX = re.compile(
    r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}|"
    r"\b(?:password|secret|apikey|token|@|\.com|\.net)\b|"
    r"\d{3,} [A-Za-z]{2,}"     # crude address start
)
API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}", re.I)
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)

load_dotenv()
genai_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ────────────────────────────────────────────────────────────────
# 1. CLI
# ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Screen capture + auto-redact (v6.4 – grouped lines, faster)"
)
parser.add_argument("-w", "--window", help="Exact window title (Windows only)")
parser.add_argument("--rtmp", help="RTMP URL (omit ⇒ local preview)")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--interval", type=int, default=1,
                    help="OCR every N frames")
parser.add_argument("--hold", type=int, default=15,
                    help="Frames to keep a box after last seen")
parser.add_argument("--pad", type=int, default=4)
parser.add_argument("--scale", type=float, default=1.0,
                    help="Scale factor before OCR (≥1 up-sample)")
parser.add_argument("--face-blur", type=int, default=75)
parser.add_argument("--color", default="255,0,0")
args = parser.parse_args()

box_color   = tuple(map(int, args.color.split(",")))
frame_delay = 1.0 / args.fps
face_k      = max(3, args.face_blur | 1)
text_k      = 35

# ────────────────────────────────────────────────────────────────
# 2. detectors
# ────────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6
)

PROMPT = (
    "You are a security assistant looking at raw OCR output.\n"
    "For any string that is *possibly* sensitive (API keys, tokens, "
    "credit-card numbers, SSNs, private or home addresses, phone numbers, "
    "emails, etc.) return its index.\n\n"
    "Keep in mind, the OCR output might be faulty and not entirely accurate of what is on the screen, so interpolate if it could be in one of those categories."
    "### Lines (index : text)\n{strings}\n\n"
    "Return a JSON array of indexes only, e.g. [1,3]."
)

# ────────────────────────────────────────────────────────────────
# 3. OCR pre-processing (lighter for speed)
# ────────────────────────────────────────────────────────────────
def preprocess(gray: np.ndarray) -> np.ndarray:
    """Mild sharpen then binary threshold – good trade-off vs speed."""
    sharp = cv2.addWeighted(gray, 1.3, cv2.GaussianBlur(gray, (0, 0), 2), -0.3, 0)
    return cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 25, 15
    )

# ────────────────────────────────────────────────────────────────
# 4. LLM helper
# ────────────────────────────────────────────────────────────────
def llm_sensitive(lines: list[str]) -> list[int]:
    prompt = PROMPT.format(strings="\n".join(f"{i}: {t}" for i, t in enumerate(lines)))
    rsp = genai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    try:
        return ast.literal_eval(rsp.text)
    except Exception:
        return []

# ────────────────────────────────────────────────────────────────
# 5. OCR wrapper (groups tokens → lines)
# ────────────────────────────────────────────────────────────────
def find_sensitive(gray: np.ndarray, scale: float) -> list[tuple[int,int,int,int]]:
    if scale != 1.0:
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR)
    data = pytesseract.image_to_data(
        preprocess(gray), output_type=pytesseract.Output.DICT,
        config=TESS_CONFIG
    )

    # group consecutive tokens that share (block, par, line)
    lines, boxes, cur_key, words, coords = [], [], None, [], []
    for i, txt in enumerate(data["text"]):
        if not txt: continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key != cur_key and words:
            lines.append(" ".join(words))
            x1 = min(c[0] for c in coords); y1 = min(c[1] for c in coords)
            x2 = max(c[2] for c in coords); y2 = max(c[3] for c in coords)
            boxes.append((x1, y1, x2, y2))
            words, coords = [], []
        cur_key = key
        words.append(txt)
        coords.append((
            data["left"][i], data["top"][i],
            data["left"][i] + data["width"][i],
            data["top"][i]  + data["height"][i]
        ))
    if words:  # flush last line
        lines.append(" ".join(words))
        x1 = min(c[0] for c in coords); y1 = min(c[1] for c in coords)
        x2 = max(c[2] for c in coords); y2 = max(c[3] for c in coords)
        boxes.append((x1, y1, x2, y2))

    if not lines:
        return []

    # quick regex gate – call LLM only if something smells sensitive
    joined = " ".join(lines)
    if not QUICK_REGEX.search(joined):
        return []

    idxs = llm_sensitive(lines)

    # backup regex scan for obvious misses
    for i, text in enumerate(lines):
        if i in idxs: continue
        if API_KEY_RE.search(text) or PHRASE_RE.search(text):
            idxs.append(i)

    if idxs:
        print("🔒 Sensitive lines:", [(i, lines[i]) for i in idxs])

    # remap boxes to original scale
    out = []
    for i in idxs:
        x1,y1,x2,y2 = boxes[i]
        if scale != 1.0:
            x1,y1,x2,y2 = [int(v / scale) for v in (x1,y1,x2,y2)]
        out.append((x1,y1,x2,y2))
    return out

# ────────────────────────────────────────────────────────────────
# 6. helper classes
# ────────────────────────────────────────────────────────────────
def expand(b, pad, W, H):
    x1,y1,x2,y2 = b
    return (max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad))

def blur(img,x1,y1,x2,y2,k):
    roi = img[y1:y2,x1:x2]
    if roi.size:
        img[y1:y2,x1:x2] = cv2.GaussianBlur(roi,(k,k),0)

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if not inter: return 0.0
    return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

class Persistent:
    def __init__(self, hold, thresh=0.5):
        self.hold=hold; self.thresh=thresh; self.items=[]
    def update(self,new):
        for it in self.items: it["ttl"]-=1
        for nb in new:
            for it in self.items:
                if iou(nb,it["box"])>=self.thresh:
                    it["box"],it["ttl"]=nb,self.hold; break
            else: self.items.append({"box":nb,"ttl":self.hold})
        self.items=[it for it in self.items if it["ttl"]>0]
    def boxes(self): return [it["box"] for it in self.items]

def pick_monitor(sct,title):
    if title and platform.system()=="Windows":
        import win32gui
        hwnd=win32gui.FindWindow(None,title)
        if not hwnd: sys.exit(f'Window "{title}" not found.')
        l,t,r,b=win32gui.GetWindowRect(hwnd)
        return {"left":l,"top":t,"width":r-l,"height":b-t}
    if title: sys.exit("Window capture by title is Windows-only.")
    return sct.monitors[0]

# ────────────────────────────────────────────────────────────────
# 7. main loop
# ────────────────────────────────────────────────────────────────
with mss() as sct, ThreadPoolExecutor(max_workers=1) as pool:
    mon = pick_monitor(sct,args.window); W,H = mon["width"],mon["height"]
    face_cache = Persistent(args.hold); text_cache = Persistent(args.hold)
    future:Future|None = None; last_txt=[]
    ffmpeg=None
    if args.rtmp:
        ffmpeg=subprocess.Popen([
            "ffmpeg","-loglevel","error","-y",
            "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}",
            "-r",str(args.fps),"-i","-","-c:v","libx264",
            "-preset","veryfast","-pix_fmt","yuv420p",
            "-f","flv",args.rtmp],stdin=subprocess.PIPE)
    frame_i=0
    try:
        while True:
            t0=time.time()
            frame=np.ascontiguousarray(np.array(sct.grab(mon))[:,:,:3])

            # faces
            faces=[]
            res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if res.detections:
                for d in res.detections:
                    bb=d.location_data.relative_bounding_box
                    faces.append((int(bb.xmin*W),int(bb.ymin*H),
                                  int((bb.xmin+bb.width)*W),
                                  int((bb.ymin+bb.height)*H)))
            face_cache.update([expand(b,args.pad,W,H) for b in faces])

            # OCR pipeline
            if frame_i%args.interval==0:
                if future and future.done():
                    try: last_txt=future.result()
                    except: last_txt=[]
                    text_cache.update([expand(b,args.pad,W,H) for b in last_txt])
                    future=None
                if future is None:
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    future=pool.submit(find_sensitive,gray,args.scale)

            for x1,y1,x2,y2 in face_cache.boxes(): blur(frame,x1,y1,x2,y2,face_k)
            for x1,y1,x2,y2 in text_cache.boxes(): blur(frame,x1,y1,x2,y2,text_k)

            cv2.rectangle(frame,(0,0),(40,40),box_color,-1)

            if ffmpeg: ffmpeg.stdin.write(frame.tobytes())
            else:
                cv2.imshow("Preview – Q quits",frame)
                if cv2.waitKey(1)&0xFF in (ord('q'),ord('Q')): break

            if (dt:=time.time()-t0)<frame_delay: time.sleep(frame_delay-dt)
            frame_i+=1
    finally:
        if ffmpeg: ffmpeg.stdin.close(); ffmpeg.wait()
        cv2.destroyAllWindows()