#!/usr/bin/env python3
"""
screen_stream_overlay.py  (v2 – with automatic redaction)

* Captures the desktop or a single window.
* Detects human faces + sensitive text (API keys, “password”, etc.).
* Blurs those regions in-line, then previews locally or streams to RTMP.
"""

import argparse, platform, subprocess, sys, time, re
from pathlib import Path
from typing import Optional

# import cv2, numpy as np
from mss import mss
import pytesseract
import mediapipe as mp
import cv2, numpy as np
import json

import cv2
import ultralytics
from google import genai
from google.genai import types
from PIL import Image
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
import base64
from letta_client import Letta

ultralytics.checks()

# Initialize the Gemini client with your API key
# client = genai.Client(api_key="sk-let-YTMwN2JlNDAtNDNhNS00MGMzLWIyYTktYjE1ZTZjZWU2ZmE0OjE4ZDFhYTY1LTdiMjMtNDNhZC04MjU5LTM0Nzk1MjdlNmVhOQ==")

client = Letta(
    token="sk-let-YTMwN2JlNDAtNDNhNS00MGMzLWIyYTktYjE1ZTZjZWU2ZmE0OjE4ZDFhYTY1LTdiMjMtNDNhZC04MjU5LTM0Nzk1MjdlNmVhOQ==",
)

# create agent
response = client.templates.agents.create(
    project="default-project",
    template_version="numerous-chocolate-mandrill:latest",
)
prompt = (
        "You are a security assistant looking at raw OCR output.\n"
        "For any string that is *definitely* sensitive (API keys, "
        "passwords, tokens, credit-card numbers, SSNs, private addresses, "
        "personal phone numbers, etc.) you must return its index. For testing purposes now, treat the literal word password as sensitive\n\n"
        "### Strings (index : text)\n" +
        "\n".join(f"{i}: {t}" for i, t in enumerate(texts)) + "\n\n"
        "Respond *only* with JSON that matches the function schema."
    )
# message agent
result = client.agents.messages.create(
    agent_id=response.agents[0].id,
    messages=[{"role": "user", "content": prompt}],
)
for message in result.messages:
    print(message)
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
PHRASE_RE  = re.compile(r"(password|secret|apikey|token|job)", re.I)
CREDIT_CARD = re.compile(r"^[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\s?")

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
def pick_monitor(sct: mss, title: Optional[str]):
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

# HELP

# def inference(image, prompt, temp=0.5):
#     """
#     Performs inference using Google Gemini 2.5 Pro Experimental model.

#     Args:
#         image (str or genai.types.Blob): The image input, either as a base64-encoded string or Blob object.
#         prompt (str): A text prompt to guide the model's response.
#         temp (float, optional): Sampling temperature for response randomness. Default is 0.5.

#     Returns:
#         str: The text response generated by the Gemini model based on the prompt and image.
#     """
#     response = client.models.generate_content(
#         model="gemini-2.5-pro-exp-03-25",
#         contents=[prompt, image],  # Provide both the text prompt and image as input
#         config=types.GenerateContentConfig(
#             temperature=temp,  # Controls creativity vs. determinism in output
#         ),
#     )

#     return response.text  # Return the generated textual response
# def inference(image, prompt, temp=0.5):
#     """
#     Performs inference using Google Gemini 2.5 Pro Experimental model.

#     Args:
#         image (np.ndarray): The image input as a NumPy array.
#         prompt (str): A text prompt to guide the model's response.
#         temp (float, optional): Sampling temperature for response randomness. Default is 0.5.

#     Returns:
#         str: The text response generated by the Gemini model based on the prompt and image.
#     """
#     # Convert the image to a base64-encoded string
#     _, buffer = cv2.imencode('.jpg', image)
#     image_base64 = base64.b64encode(buffer).decode('utf-8')

#     response = client.models.generate_content(
#         model="gemini-2.5-pro-exp-03-25",
#         contents=[prompt, image_base64],  # Provide both the text prompt and image as input
#         config=types.GenerateContentConfig(
#             temperature=temp,  # Controls creativity vs. determinism in output
#         ),
#     )
#     return response.text  # Return the generated textual response
# def read_image(filename=None):
#     if filename is not None:
#         image_name = filename
#     else:
#         image_name = "bus.jpg"  # or "zidane.jpg"

#     # Download the image
#     safe_download(f"https://github.com/ultralytics/notebooks/releases/download/v0.0.0/{image_name}")

#     # Read image with opencv
#     image = cv2.cvtColor(cv2.imread(f"/content/{image_name}"), cv2.COLOR_BGR2RGB)

#     # Extract width and height
#     h, w = image.shape[:2]

#     # # Read the image using OpenCV and convert it into the PIL format
#     return Image.fromarray(image), w, h

# def clean_results(results):
#     """Clean the results for visualization."""
#     return results.strip().removeprefix("```json").removesuffix("```").strip()


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

    # img_detection = False
    try:
        while True:
            t0 = time.time()

            # capture
            frame = sct.grab(monitor)
            img = np.ascontiguousarray(np.array(frame)[:, :, :3])

            # -------------------------------------------------- #
            # OBJECT DETECTION
            
            # if not img_detection:
            #     # Define the text prompt
            #     prompt = """
            #     Detect the 2d bounding boxes of objects in image.
            #     """

            #     # Fixed, plotting function depends on this.
            #     output_prompt = "Return just box_2d and labels, no additional text."
                
            #     height = img.shape[0]  # Get the height
            #     width = img.shape[1]  
            #     image = img
            #     # image, w, h = read_image("gemini-image1.jpg")  # Read img, extract width, height

            #     results = inference(image, prompt + output_prompt)  # Perform inference

            #     cln_results = json.loads(clean_results(results))  # Clean results, list convert

            #     annotator = Annotator(image)  # initialize Ultralytics annotator

            #     for idx, item in enumerate(cln_results):
            #         # By default, gemini model return output with y coordinates first.
            #         # Scale normalized box coordinates (0–1000) to image dimensions
            #         y1, x1, y2, x2 = item["box_2d"]  # bbox post processing,
            #         y1 = y1 / 1000 * h
            #         x1 = x1 / 1000 * w
            #         y2 = y2 / 1000 * h
            #         x2 = x2 / 1000 * w

            #         if x1 > x2:
            #             x1, x2 = x2, x1  # Swap x-coordinates if needed
            #         if y1 > y2:
            #             y1, y2 = y2, y1  # Swap y-coordinates if needed

            #         annotator.box_label([x1, y1, x2, y2], label=item["label"], color=colors(idx, True))

            #     # Image.fromarray(annotator.result())  # display the output
            #     img = Image.fromarray(annotator.result())
            # img_detection = True
            ## END HELP

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
