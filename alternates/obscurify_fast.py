#!/usr/bin/env python3
"""
screen_stream_overlay_async.py  (v7.11 – detailed prompt, fixed filters)

Workflow
========
1. A Tkinter window appears first with 7 check-boxes:
     • Passwords / API keys
     • Emails
     • Phone numbers
     • Credit-card numbers
     • Addresses
     • Names
     • Faces
2. Click **Start** — the dashboard closes and capture begins.
3. Only checked categories are blurred. Unchecked categories are guaranteed
   to remain visible because the Gemini prompt forbids blurring them.
"""

from __future__ import annotations
import argparse, platform, subprocess, sys, time, re, os, json, ast
from concurrent.futures import ThreadPoolExecutor, Future
import tkinter as tk

import cv2, numpy as np
from mss import mss
import pytesseract, mediapipe as mp
from dotenv import load_dotenv
import google.generativeai as genai

# ───────────── OCR & regex helpers ──────────────────────────────────────────
TESS_PATH   = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESS_PATH
TESS_CONFIG = "--oem 3 --psm 11 -l eng"

API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}", re.I)
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)
EMAIL_RE   = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,}")

# ───────────── Gemini setup ────────────────────────────────────────────────
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ───────────── dashboard (blocking) ────────────────────────────────────────
CATS = {
    "Passwords / API keys": "pw",
    "Emails":               "email",
    "Phone numbers":        "phone",
    "Credit-card numbers":  "cc",
    "Addresses":            "addr",
    "Names":                "name",
    "Faces":                "face",
}

def choose_filters() -> set[str]:
    root = tk.Tk()
    root.title("Select categories to BLUR")
    vars_: dict[str, tk.BooleanVar] = {}
    tk.Label(root, text="Blur the checked categories:", font=("Arial", 12, "bold")).pack()
    for label in CATS:
        var = tk.BooleanVar(value=True)
        vars_[label] = var
        tk.Checkbutton(root, text=label, variable=var, anchor="w").pack(fill="x")
    selected: set[str] = set()
    def _start():
        nonlocal selected
        selected = {CATS[lbl] for lbl, v in vars_.items() if v.get()}
        root.destroy()
    tk.Button(root, text="Start", command=_start, width=10).pack(pady=8)
    root.mainloop()
    return selected

FILTER_CODES = choose_filters()
print("Blurring:", FILTER_CODES or "nothing (all text visible)")

# ───────────── prompt builder ───────────────────────────────────────────────
_BULLETS = {
    "pw":   "• Passwords, passphrases, API tokens, secrets",
    "email":"• E-mail addresses",
    "phone":"• Personal phone numbers (7–15 digits, any locale)",
    "cc":   "• Credit-card or bank numbers (≈12–19 digits, with/without dashes)",
    "addr": "• Street / mailing addresses (number + street + city/state/zip)",
    "name": "• Personal names **when paired with other private data** on the line",
}
_DESC = {
    "pw": "passwords/API keys", "email": "emails", "phone": "phone numbers",
    "cc": "credit-card numbers", "addr": "addresses", "name": "names"
}

def make_prompt(lines: list[str]) -> str:
    checked  = FILTER_CODES & _BULLETS.keys()
    unchecked = (set(_BULLETS) - checked)
    bullets = "\n".join(_BULLETS[c] for c in checked) or "• (none — blur no text categories)"
    unchecked_list = ", ".join(_DESC[c] for c in unchecked) or "none"

    return f"""You are a security assistant reading raw OCR lines from a live
screen.  The text may contain minor recognition errors (e.g. '8' instead of 'B').
Mentally correct obvious typos, but **only blur** a line if you are at least
≈80 % certain it contains sensitive data belonging to the *checked* categories
below.  

If a line matches an **unchecked** category (e.g. {unchecked_list}), you must
**leave it visible** — do not blur it.

Checked categories to BLUR:
{bullets}

Ignore generic tech terms like “login”, “user”, “id”, etc. unless strong evidence
shows an actual secret (digits, token patterns, etc.).

If no line needs blurring, return [].

Lines (index : text):
{chr(10).join(f"{i}: {t}" for i, t in enumerate(lines))}

Respond with JSON only, e.g. [1,3]  — no extra commentary."""

# ───────────── CLI / capture parameters ─────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-w","--window"); ap.add_argument("--rtmp")
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--interval", type=int, default=1)
ap.add_argument("--hold", type=int, default=40); ap.add_argument("--pad", type=int, default=4)
ap.add_argument("--scale", type=float, default=1.5); ap.add_argument("--face-blur", type=int, default=75)
ap.add_argument("--color", default="255,0,0")
args = ap.parse_args()

box_color   = tuple(map(int,args.color.split(",")))
frame_delay = 1/args.fps
face_k      = max(3, args.face_blur | 1)
text_k      = 35
mp_face = mp.solutions.face_detection.FaceDetection(0,0.6) if "face" in FILTER_CODES else None

# ───────────── helpers ─────────────────────────────────────────────────────
def preprocess(gray):
    sharp = cv2.addWeighted(gray,1.5,cv2.GaussianBlur(gray,(0,0),1.2),-0.5,0)
    bw = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,35,11)
    return cv2.morphologyEx(bw,cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),1)

def robust_json_extract(txt:str)->list[int]:
    try:
        out = ast.literal_eval(txt.strip())
        return [int(x) for x in out] if isinstance(out,list) else []
    except Exception:
        m = re.search(r"\[[^\]]*\]", txt)
        if m:
            try: return [int(x) for x in json.loads(m.group())]
            except: pass
    return []

def llm_sensitive(lines):
    if not (FILTER_CODES & _BULLETS.keys()):
        return []
    prompt = make_prompt(lines)
    try:
        rsp = model.generate_content(prompt, stream=False)
        body = rsp.text if hasattr(rsp,"text") else rsp.candidates[0].content.parts[0].text
        return robust_json_extract(body)
    except Exception as e:
        print("[Gemini error]", e); return []

def find_sensitive(gray,scale):
    if scale!=1.0:
        gray = cv2.resize(gray,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    d=pytesseract.image_to_data(preprocess(gray),
        output_type=pytesseract.Output.DICT,config=TESS_CONFIG)

    lines,boxes,cur,words,coords = [],[],None,[],[]
    for i,txt in enumerate(d["text"]):
        if not txt: continue
        key=(d["block_num"][i],d["par_num"][i],d["line_num"][i])
        if key!=cur and words:
            xs,ys,x2s,y2s=zip(*coords)
            lines.append(" ".join(words))
            boxes.append((min(xs),min(ys),max(x2s),max(y2s)))
            words,coords=[],[]
        cur=key; words.append(txt)
        coords.append((d["left"][i],d["top"][i],
                       d["left"][i]+d["width"][i],
                       d["top"][i]+d["height"][i]))
    if words:
        xs,ys,x2s,y2s=zip(*coords)
        lines.append(" ".join(words))
        boxes.append((min(xs),min(ys),max(x2s),max(y2s)))

    if not lines: return []

    idxs=set(llm_sensitive(lines))
    for i,t in enumerate(lines):
        if i in idxs: continue
        if "pw" in FILTER_CODES and (API_KEY_RE.search(t) or PHRASE_RE.search(t)):
            idxs.add(i)
        elif "email" in FILTER_CODES and EMAIL_RE.search(t):
            idxs.add(i)

    out=[]
    for i in idxs:
        x1,y1,x2,y2 = boxes[i]
        if scale!=1.0:
            x1,y1,x2,y2 = [int(v/scale) for v in (x1,y1,x2,y2)]
        out.append((x1,y1,x2,y2))
    return out

def expand(b,p,W,H):
    x1,y1,x2,y2=b
    return (max(0,x1-p),max(0,y1-p),min(W,x2+p),min(H,y2+p))

def blur(img,x1,y1,x2,y2,k):
    roi = img[y1:y2,x1:x2]
    if roi.size: img[y1:y2,x1:x2]=cv2.GaussianBlur(roi,(k,k),0)

class Hold:
    def __init__(self,ttl): self.ttl=ttl; self.items=[]
    def update(self,new):
        for it in self.items: it["ttl"]-=1
        for b in new: self.items.append({"box":b,"ttl":self.ttl})
        self.items = [it for it in self.items if it["ttl"]>0]
    def boxes(self): return [it["box"] for it in self.items]

def pick_monitor(sct,title):
    if title and platform.system()=="Windows":
        import win32gui
        hwnd=win32gui.FindWindow(None,title)
        if not hwnd: sys.exit(f'"{title}" not found')
        l,t,r,b=win32gui.GetWindowRect(hwnd)
        return {"left":l,"top":t,"width":r-l,"height":b-t}
    if title: sys.exit("Window capture by title is Windows-only")
    return sct.monitors[0]

# ───────────── main loop ───────────────────────────────────────────────────
with mss() as sct, ThreadPoolExecutor(max_workers=1) as pool:
    mon = pick_monitor(sct,args.window); W,H=mon["width"],mon["height"]
    face_hold = Hold(args.hold); text_hold = Hold(args.hold)
    future: Future|None = None; last_txt=[]
    ffmpeg=None
    if args.rtmp:
        ffmpeg=subprocess.Popen([
            "ffmpeg","-loglevel","error","-y","-f","rawvideo",
            "-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(args.fps),
            "-i","-","-c:v","libx264","-preset","veryfast",
            "-pix_fmt","yuv420p","-f","flv",args.rtmp],stdin=subprocess.PIPE)

    frame_i = 0
    try:
        while True:
            t0=time.time()
            frame = np.ascontiguousarray(np.array(sct.grab(mon))[:,:,:3])

            # faces
            if mp_face:
                faces=[]
                res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if res.detections:
                    for d in res.detections:
                        bb=d.location_data.relative_bounding_box
                        faces.append((int(bb.xmin*W),int(bb.ymin*H),
                                      int((bb.xmin+bb.width)*W),
                                      int((bb.ymin+bb.height)*H)))
                face_hold.update([expand(b,args.pad,W,H) for b in faces])

            # OCR async
            if frame_i%args.interval==0:
                if future and future.done():
                    try: last_txt=future.result()
                    except: last_txt=[]
                    text_hold.update([expand(b,args.pad,W,H) for b in last_txt])
                    future=None
                if future is None and (FILTER_CODES & set(_BULLETS)):
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    future=pool.submit(find_sensitive,gray,args.scale)

            # draw
            for x1,y1,x2,y2 in face_hold.boxes(): blur(frame,x1,y1,x2,y2,face_k)
            for x1,y1,x2,y2 in text_hold.boxes(): blur(frame,x1,y1,x2,y2,text_k)
            cv2.rectangle(frame,(0,0),(40,40),box_color,-1)

            if ffmpeg: ffmpeg.stdin.write(frame.tobytes())
            else:
                cv2.imshow("Preview – Q quit",frame)
                if cv2.waitKey(1)&0xFF in (ord("q"),ord("Q")): break

            dt=time.time()-t0
            if dt<frame_delay: time.sleep(frame_delay-dt)
            frame_i+=1
    finally:
        if ffmpeg: ffmpeg.stdin.close(); ffmpeg.wait()
        cv2.destroyAllWindows()
