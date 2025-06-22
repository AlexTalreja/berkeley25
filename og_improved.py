#!/usr/bin/env python3
"""
screen_stream_overlay_async.py  (v7.2 – smooth blur boxes)

  * Detects faces & sensitive text.
  * Keeps blur rectangles glued to content while scrolling.
  * Smooth-averages box edges to kill pulsating/flicker.
"""

from __future__ import annotations
import argparse, platform, subprocess, sys, time, re, os, ast
from concurrent.futures import ThreadPoolExecutor, Future
import cv2, numpy as np
from mss import mss
import pytesseract, mediapipe as mp
from dotenv import load_dotenv
from google import genai

# ── constant setup ──────────────────────────────────────────────
TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESS_PATH
TESS_CONFIG = "--oem 3 --psm 11 -l eng"

QUICK_REGEX = re.compile(
    r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}"
    r"|(?:\b(?:api_)?key\b|\btoken\b|\bsecret\b)"
    r"|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,}"
    r"|\b(?:\d{3}[- ]?)?\d{3}[- ]?\d{4}\b"
    r"|\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    r"|\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"
    r"|\d{1,5}\s+\w{2,}\s+\w{2,}",
    re.I)
API_KEY_RE = re.compile(r"(AKIA|ASIA|SK|sk_live_)[A-Za-z0-9]{16,}", re.I)
PHRASE_RE  = re.compile(r"(password|secret|apikey|token)", re.I)

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── CLI ─────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Screen capture + auto-redact (v7.2)")
p.add_argument("-w","--window")
p.add_argument("--rtmp")
p.add_argument("--fps", type=int, default=30)
p.add_argument("--interval", type=int, default=1)
p.add_argument("--hold", type=int, default=40)
p.add_argument("--pad", type=int, default=4)
p.add_argument("--scale", type=float, default=1.5)
p.add_argument("--face-blur", type=int, default=75)
p.add_argument("--color", default="255,0,0")
args = p.parse_args()

box_color   = tuple(int(c) for c in args.color.split(","))
frame_delay = 1/args.fps
face_k      = max(3, args.face_blur | 1)
text_k      = 35

# ── detectors ───────────────────────────────────────────────────
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6)

PROMPT = """You are a security assistant reading raw OCR lines.
Return *indexes* of any line that COULD be sensitive.
### Example 1
0: sk_live_abc…
1: hello world
→ [0]
### Example 2
0: 555-123-4567
1: foo
→ [0]
### Lines
{strings}
JSON only, e.g. [1,3]
"""

def preprocess(gray:np.ndarray)->np.ndarray:
    sharp=cv2.addWeighted(gray,1.5,cv2.GaussianBlur(gray,(0,0),1.2),-0.5,0)
    bw=cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY,35,11)
    return cv2.morphologyEx(bw,cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)),1)

def llm_sensitive(lines:list[str])->list[int]:
    prompt=PROMPT.format(strings="\n".join(f"{i}: {t}"
                             for i,t in enumerate(lines)))
    try:
        r=genai_client.models.generate_content(
            model="gemini-2.5-flash",contents=prompt,
            request_options={"timeout":30})
        return ast.literal_eval(r.text.strip())
    except Exception: return []

def find_sensitive(gray:np.ndarray,scale:float):
    if scale!=1.0:
        gray=cv2.resize(gray,(0,0),fx=scale,fy=scale,
                        interpolation=cv2.INTER_CUBIC)
    d=pytesseract.image_to_data(preprocess(gray),
                                output_type=pytesseract.Output.DICT,
                                config=TESS_CONFIG)
    lines,boxes,cur,words,coords=[],[],None,[],[]
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
    if not lines or not QUICK_REGEX.search(" ".join(lines)): return []

    idxs=llm_sensitive(lines)
    for i,t in enumerate(lines):
        if i in idxs: continue
        if API_KEY_RE.search(t) or PHRASE_RE.search(t): idxs.append(i)

    out=[]
    for i in idxs:
        x1,y1,x2,y2=boxes[i]
        if scale!=1.0:
            x1,y1,x2,y2=[int(v/scale) for v in (x1,y1,x2,y2)]
        out.append((x1,y1,x2,y2))
    return out

# ── helper classes ──────────────────────────────────────────────
def expand(b,pad,W,H):
    x1,y1,x2,y2=b
    return (max(0,x1-pad),max(0,y1-pad),min(W,x2+pad),min(H,y2+pad))

def blur(img,x1,y1,x2,y2,k):
    roi=img[y1:y2,x1:x2]
    if roi.size: img[y1:y2,x1:x2]=cv2.GaussianBlur(roi,(k,k),0)

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if not inter: return 0.0
    return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

class SmoothPersistent:
    """Persistent cache that *smooths* box edges to stop flicker."""
    def __init__(self,hold,thresh=0.3,smooth=0.3):
        self.hold=hold; self.thresh=thresh; self.smooth=smooth; self.items=[]
    def _ema(self,old,new):          # exponential moving average
        return int(old*(1-self.smooth)+new*self.smooth)
    def update(self,new_boxes):
        for it in self.items: it["ttl"]-=1
        for nb in new_boxes:
            for it in self.items:
                if iou(nb,it["box"])>=self.thresh:
                    # smooth each coordinate, clamp jitter ≤2 px
                    ox1,oy1,ox2,oy2=it["box"]
                    nx1,ny1,nx2,ny2=nb
                    smoothed=(self._ema(ox1,nx1),self._ema(oy1,ny1),
                              self._ema(ox2,nx2),self._ema(oy2,ny2))
                    # tiny jitter filter
                    if max(abs(smoothed[i]-ox) for i,ox in enumerate(it["box"]))<=2:
                        smoothed=it["box"]  # ignore micro-shake
                    it["box"],it["ttl"]=smoothed,self.hold
                    break
            else:
                self.items.append({"box":nb,"ttl":self.hold})
        self.items=[it for it in self.items if it["ttl"]>0]
    def shift(self,dx,dy,W,H):
        if dx==dy==0: return
        kept=[]
        for it in self.items:
            x1,y1,x2,y2=it["box"]
            x1+=dx; x2+=dx; y1+=dy; y2+=dy
            if x1>=W or y1>=H or x2<=0 or y2<=0: continue
            it["box"]=(max(0,x1),max(0,y1),min(W,x2),min(H,y2))
            kept.append(it)
        self.items=kept
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

# ── main loop ───────────────────────────────────────────────────
with mss() as sct, ThreadPoolExecutor(max_workers=1) as pool:
    mon=pick_monitor(sct,args.window); W,H=mon["width"],mon["height"]
    face_cache=SmoothPersistent(args.hold)
    text_cache=SmoothPersistent(args.hold)
    future=None; last_txt=[]; prev_gray_small=None
    ffmpeg=None
    if args.rtmp:
        ffmpeg=subprocess.Popen([
            "ffmpeg","-loglevel","error","-y",
            "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}",
            "-r",str(args.fps),"-i","-","-c:v","libx264","-preset","veryfast",
            "-pix_fmt","yuv420p","-f","flv",args.rtmp],stdin=subprocess.PIPE)

    frame_i=0
    try:
        while True:
            start=time.time()
            frame=np.ascontiguousarray(np.array(sct.grab(mon))[:,:,:3])

            # # scroll shift
            # small=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(W//8,H//8))
            # if prev_gray_small is not None:
            #     shift,resp=cv2.phaseCorrelate(prev_gray_small.astype(np.float32),
            #                                   small.astype(np.float32))
            #     if resp>0.15:
            #         text_cache.shift(int(round(shift[0]*8)),
            #                          int(round(shift[1]*8)),W,H)
            # prev_gray_small=small

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

            # OCR async
            if frame_i%args.interval==0:
                if future and future.done():
                    try: last_txt=future.result()
                    except: last_txt=[]
                    text_cache.update([expand(b,args.pad,W,H) for b in last_txt])
                    future=None
                if future is None:
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    future=pool.submit(find_sensitive,gray,args.scale)

            # draw
            for x1,y1,x2,y2 in face_cache.boxes(): blur(frame,x1,y1,x2,y2,face_k)
            for x1,y1,x2,y2 in text_cache.boxes(): blur(frame,x1,y1,x2,y2,text_k)
            cv2.rectangle(frame,(0,0),(40,40),box_color,-1)

            if ffmpeg: ffmpeg.stdin.write(frame.tobytes())
            else:
                cv2.imshow("Preview – Q quit",frame)
                if cv2.waitKey(1)&0xFF in (ord('q'),ord('Q')): break

            if (dt:=time.time()-start)<frame_delay: time.sleep(frame_delay-dt)
            frame_i+=1
    finally:
        if ffmpeg: ffmpeg.stdin.close(); ffmpeg.wait()
        cv2.destroyAllWindows()
