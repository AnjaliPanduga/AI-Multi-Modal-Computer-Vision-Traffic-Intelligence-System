
import io
import os
import cv2
import time
import queue
import threading
import numpy as np
import tempfile
import requests
import streamlit as st
import plotly.graph_objects as go
from collections import deque
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import sqlite3
from datetime import datetime

from sklearn.linear_model import LinearRegression
start_time = time.time()



try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Smart Object Detection v2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# PREMIUM DARK UI CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    margin: 6px 0;
    box-shadow: 0 4px 20px rgba(37,99,235,0.1);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card h2 { color: #60a5fa; font-size: 2.2rem; margin: 0; font-weight: 800; }
.metric-card p  { color: #64748b; margin: 4px 0 0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* Banner */
.banner {
    background: linear-gradient(135deg, #0f1f40 0%, #0a0e1a 40%, #1a0f2e 100%);
    border: 1px solid #2563eb33;
    border-radius: 16px;
    padding: 22px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, #2563eb15 0%, transparent 60%);
    pointer-events: none;
}
.banner h1 { font-size: 2rem; margin: 0; color: #93c5fd; font-weight: 800; }
.banner p  { color: #64748b; margin: 6px 0 0; font-size: 0.95rem; }
.banner .badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 10px;
    margin-right: 6px;
    border: 1px solid #2563eb44;
}

/* Info panel */
.info-box {
    background: linear-gradient(135deg, #0f1923 0%, #0a1628 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* Emotion badges */
.emotion-happy    { color: #34d399; font-weight: 700; }
.emotion-sad      { color: #60a5fa; font-weight: 700; }
.emotion-angry    { color: #f87171; font-weight: 700; }
.emotion-surprised{ color: #fbbf24; font-weight: 700; }
.emotion-neutral  { color: #94a3b8; font-weight: 700; }

/* Feature badge */
.feature-pill {
    display: inline-block;
    background: #1a2744;
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 12px;
    margin: 2px 2px;
    border: 1px solid #2563eb33;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
    transition: all 0.25s;
    letter-spacing: 0.03em;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px #2563eb55;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border-color: #1e3a5f !important;
    border-radius: 10px !important;
}

/* Slider */
.stSlider > div > div { background: #1e3a5f; }
.stSlider > div > div > div { background: #3b82f6; }

/* Progress */
.stProgress > div > div > div { background: linear-gradient(90deg, #2563eb, #7c3aed); }

/* Divider */
hr { border-color: #1e2d45 !important; }

/* Chart container */
.chart-container {
    background: #0f1629;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 16px;
    margin: 12px 0;
}

/* Snapshot overlay */
.snap-card {
    background: linear-gradient(135deg, #1a2744, #0f172a);
    border: 1px solid #2563eb55;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 12px 0;
}

/* Heatmap label */
.heatmap-title {
    text-align: center;
    color: #60a5fa;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AUTO-DOWNLOAD MISSING CASCADES
# ─────────────────────────────────────────────────────────────
CASCADE_URLS = {
    "haarcascade_car.xml": [
        "https://raw.githubusercontent.com/souravdeyone/OpenCV-Reference/master/Haarcascades/haarcascade_car.xml",
        "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml",
    ],
    "haarcascade_fullbody.xml": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_fullbody.xml",
        "https://raw.githubusercontent.com/npinto/opencv/master/data/haarcascades/haarcascade_fullbody.xml",
    ],
}

def ensure_cascade(filename: str) -> str:
    """Return local path, downloading the cascade from mirrors if missing."""
    local = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(local):
        urls = CASCADE_URLS.get(filename, [])
        for url in urls:
            try:
                with st.spinner(f"Downloading {filename} …"):
                    r = requests.get(url, timeout=20)
                    r.raise_for_status()
                    with open(local, "wb") as f:
                        f.write(r.content)
                break   # success
            except Exception:
                continue  # try next mirror
    return local

# Load classifiers
face_cls  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cls   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cls = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
car_cls   = cv2.CascadeClassifier(ensure_cascade("haarcascade_car.xml"))
body_cls  = cv2.CascadeClassifier(ensure_cascade("haarcascade_fullbody.xml"))

# HOG people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# YOLOv8 Vehicle Detection Model
yolo_model = YOLO("yolov8n.pt")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────


# SQLite Database Connection
conn = sqlite3.connect("traffic_analytics.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    frame INTEGER,
    vehicle_count INTEGER,
    crossed_count INTEGER
)
""")

conn.commit()

log_data = pd.DataFrame(columns=[
    "Time",
    "Frame_Number",
    "Vehicle_Count",
    "Crossed_Count"
])

MODES = {
    "🔍 Face Detection":              "face",
    "👁️ Face & Eye Detection":        "face_eye",
    "🎭 Emotion Detection":           "emotion",
    "🚗 Car Detection (Video)":       "car",
    "🚶 Full Body — Haar (Video)":    "body",
    "🕵️ HOG People Detector (Video)": "hog",
    "🎥 Live Webcam Detection":       "webcam",
}

with st.sidebar:
    st.markdown("## 🎯 Detection Mode")
    mode_label = st.selectbox("Select mode", list(MODES.keys()), label_visibility="collapsed")
    mode = MODES[mode_label]

    st.markdown("---")

    # ── Mode-specific settings ──────────────────────────────
    st.markdown("### ⚙️ Settings")

    anonymize_faces = False
    frame_skip = 1
    show_mask = True
    show_heatmap = True
    show_trails = True
    show_chart = True

    if mode in ("face", "face_eye", "emotion", "webcam"):
        anonymize_faces = st.checkbox("🔒 Anonymize Faces (Blur)", value=False)

    if mode in ("car", "body", "hog"):
        frame_skip = st.slider("⚡ Process every N frames", 1, 5, 1,
                               help="Higher = faster but skips frames")
        show_chart = st.checkbox("📊 Show Live Stats Chart", value=True)

    if mode == "car":
        show_mask   = st.checkbox("🟢 Car Outline View (side-by-side)", value=True)
        show_trails = st.checkbox("🌊 Motion Trails", value=True)

    if mode == "body":
        show_heatmap = st.checkbox("🌡️ Crowd Density Heatmap", value=True)

    st.markdown("---")
    st.markdown("### ✨ Features")
    features = [
        "Multi-face & eye detection", "Emotion recognition",
        "HOG people detector", "Face anonymizer",
        "Crowd heatmap", "Motion trails",
        "Car line-counter", "Plate detection",
        "KM estimation", "Live chart",
        "Webcam snapshot", "FPS counter",
    ]
    pills_html = "".join(f'<span class="feature-pill">{f}</span>' for f in features)
    st.markdown(f'<div style="line-height:2">{pills_html}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with OpenCV + Streamlit  |  v2.0")

# ─────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="banner">
  <h1>🚀 AI-Based Multi-Modal Computer Vision & Traffic Intelligence System <span style="font-size:1rem;color:#64748b;font-weight:400">v2.0</span></h1>
  <p>Current mode: <strong style="color:#93c5fd">{mode_label}</strong></p>
  <div>
    <span class="badge">OpenCV</span>
    <span class="badge">Haar Cascades</span>
    <span class="badge">HOG+SVM</span>
    <span class="badge">Real-Time</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER — Pixelate / blur a face ROI
# ─────────────────────────────────────────────────────────────
def anonymize_roi(img: np.ndarray, x: int, y: int, w: int, h: int,
                  blocks: int = 12) -> np.ndarray:
    """Pixelate the face region for privacy."""
    roi = img[y:y+h, x:x+w]
    small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

# ─────────────────────────────────────────────────────────────
# HELPER — Accurate Multi-Signal Emotion Recognizer
# Signals: smile cascade · eye openness · forehead tension
#          region intensity · edge density · LBP-like texture
# Labels : Happy · Sad · Angry · Surprised · Fear · Disgust · Neutral
# ─────────────────────────────────────────────────────────────
FER_LABELS = ["Happy", "Sad", "Angry", "Surprised", "Fear", "Disgust", "Neutral"]

EMOTION_COLORS = {
    "Happy":     (0, 230, 130),
    "Surprise":  (0, 210, 255),
    "Surprised": (0, 210, 255),
    "Angry":     (60, 60, 255),
    "Fear":      (180, 80, 255),
    "Sad":       (255, 160, 50),
    "Disgust":   (0, 180, 100),
    "Neutral":   (180, 180, 180),
}

_emotion_model_path = os.path.join(os.path.dirname(__file__), "emotion_model.onnx")
_emotion_net = None

def _load_onnx_net():
    """Try to load the ONNX FER model once (lazy, safe)."""
    global _emotion_net
    if _emotion_net is not None:
        return _emotion_net
    # Only use if we verified it actually works as an emotion model
    # (the OpenCV Zoo model was misidentified — skip it)
    return None

def estimate_emotion(face_bgr: np.ndarray) -> tuple:
    """
    Accurate multi-signal emotion estimation.
    Combines smile cascade, eye openness, forehead edge density,
    mouth region analysis, and Laplacian texture to detect:
    Happy · Sad · Angry · Surprised · Fear · Neutral
    Returns (emotion_label, confidence_0_to_1)
    """
    if face_bgr is None or face_bgr.size == 0:
        return "Neutral", 0.5

    h, w = face_bgr.shape[:2]
    if h < 30 or w < 30:
        return "Neutral", 0.5

    # ── Pre-process ──────────────────────────────────────────
    face = cv2.resize(face_bgr, (96, 96))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # ── Region definitions ───────────────────────────────────
    forehead  = gray_eq[0:25,  :]       # top 26% — brows
    eye_zone  = gray_eq[18:48, :]       # eyes band
    nose_zone = gray_eq[40:65, :]       # nose
    mouth_z   = gray_eq[60:90, :]       # mouth/chin
    upper_h   = gray_eq[:48,   :]       # top half

    mean_mouth   = float(np.mean(mouth_z))
    mean_eye     = float(np.mean(eye_zone))
    mean_overall = float(np.mean(gray_eq))
    nose_mean    = float(np.mean(nose_zone)) if nose_zone.size > 0 else mean_overall

    # ── Signal 1: Smile detection  ───────────────────────────
    # Lower minNeighbors so real smiles are not missed
    smiles = smile_cls.detectMultiScale(
        gray_eq, scaleFactor=1.3, minNeighbors=8, minSize=(15, 6)
    )
    smile_score = float(min(len(smiles) * 0.5, 1.0))

    # ── Signal 2: Eye openness ───────────────────────────────
    eyes = eye_cls.detectMultiScale(
        eye_zone, scaleFactor=1.1, minNeighbors=4, minSize=(6, 6)
    )
    eye_count = len(eyes)

    # ── Signal 3: Forehead edge density (raised eyebrows) ────
    fore_edges   = cv2.Canny(forehead, 25, 75)
    edge_density = float(np.mean(fore_edges > 0))

    # ── Signal 4: Laplacian texture (brow furrowing) ─────────
    upper_lap      = cv2.Laplacian(upper_h, cv2.CV_64F)
    upper_sharpness = float(np.var(upper_lap))

    # ── Signal 5: Mouth vs eye brightness ratio ───────────────
    # Positive → mouth corners down (sad/angry)
    mouth_vs_eye_diff = mean_eye - mean_mouth

    # ── Signal 6: Overall brightness ─────────────────────────
    bright_face = mean_overall > 130   # well-lit / happy face
    dark_face   = mean_overall < 100   # darker / sad/angry face

    # ── Score each emotion (LOW Neutral baseline so others can win) ──
    scores = {
        "Happy":     0.0,
        "Sad":       0.0,
        "Angry":     0.0,
        "Surprised": 0.0,
        "Fear":      0.0,
        "Neutral":   0.30,   # Reduced baseline — others can easily beat this
    }

    # ── HAPPY: smile is the dominant, most reliable signal ───
    if smile_score > 0:
        scores["Happy"] += smile_score * 2.5
    if bright_face:
        scores["Happy"] += 0.3
    # Wide open mouth area (brighter) also suggests smile
    if mean_mouth > 145:
        scores["Happy"] += 0.4

    # ── SURPRISED: raised brows (edge density) + wide eyes ───
    if edge_density > 0.10:
        scores["Surprised"] += (edge_density - 0.10) * 5.5
        # Wide open eyes are a secondary signal — only useful WITH raised brows
        if eye_count >= 1:
            scores["Surprised"] += eye_count * 0.20
    # Surprised usually has open mouth (brighter mouth region) — minor bonus
    if edge_density > 0.10 and mean_mouth > 135:
        scores["Surprised"] += 0.20

    # ── ANGRY: brow furrowing (high Laplacian) + darker face ─
    if upper_sharpness > 1500:
        scores["Angry"] += min((upper_sharpness - 1500) / 4000.0, 1.0) * 1.5
    if dark_face:
        scores["Angry"] += 0.4
    # Downturned corners (mouth darker than eyes)
    if mouth_vs_eye_diff > 15:
        scores["Angry"] += min((mouth_vs_eye_diff - 15) / 60.0, 1.0) * 0.8
    # If smile → suppress angry
    if smile_score > 0.3:
        scores["Angry"] *= 0.1

    # ── SAD: mouth distinctly darker + low energy face ────────
    if mouth_vs_eye_diff > 12 and smile_score < 0.1:
        scores["Sad"] += min((mouth_vs_eye_diff - 12) / 70.0, 1.0) * 2.0
    if dark_face and smile_score < 0.1:
        scores["Sad"] += 0.4
    if edge_density < 0.06:   # flat brows = dejected
        scores["Sad"] += 0.3

    # ── FEAR: raised brows + dark face ────────────────────────
    if edge_density > 0.12 and dark_face:
        scores["Fear"] += (edge_density - 0.12) * 3.0
        scores["Fear"] += (100 - mean_overall) / 100.0 * 0.5

    # ── Mutual inhibition ─────────────────────────────────────
    if scores["Happy"] > 0.5:
        scores["Sad"]  *= 0.05
        scores["Angry"] *= 0.1
    if scores["Angry"] > 0.8:
        scores["Happy"] *= 0.1

    # Remove negatives
    for k in scores:
        scores[k] = max(0.0, scores[k])

    # Normalize
    total = sum(scores.values()) + 1e-6
    probs = {k: v / total for k, v in scores.items()}

    best_emotion = max(probs, key=probs.get)
    confidence   = probs[best_emotion]

    return best_emotion, min(confidence * 1.15, 0.95)

# ─────────────────────────────────────────────────────────────
# HELPER — Draw all faces + eyes on an image
# ─────────────────────────────────────────────────────────────
def detect_faces_eyes(img: np.ndarray, draw_eyes: bool = True,
                      anon: bool = False) -> tuple:
    """Returns annotated BGR image, face_count, eye_count."""
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cls.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    total_eyes = 0

    for i, (x, y, w, h) in enumerate(faces):
        if anon:
            anonymize_roi(img, x, y, w, h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 255), 2)
            cv2.putText(img, "Anonymized", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 1)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 80, 80), 2)
            cv2.putText(img, f"Face {i+1}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 180), 2)

            if draw_eyes:
                eye_y_end = int(y + h * 0.60)
                roi_gray  = gray[y:eye_y_end, x:x+w]
                roi_color = img [y:eye_y_end, x:x+w]

                eyes = eye_cls.detectMultiScale(
                    roi_gray, scaleFactor=1.08, minNeighbors=6,
                    minSize=(12, 12), maxSize=(int(w*0.45), int(h*0.35)),
                )
                valid_eyes = sorted(
                    [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes
                     if ew >= 12 and ey <= roi_gray.shape[0]*0.80],
                    key=lambda e: e[0]
                )[:2]
                total_eyes += len(valid_eyes)
                for (ex, ey, ew, eh) in valid_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 230, 100), 2)

    hud = f"Faces: {len(faces)}  |  Eyes: {total_eyes}"
    cv2.rectangle(img, (0, 0), (320, 44), (0, 0, 0), -1)
    cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 200, 255), 2)

    return img, len(faces), total_eyes

# ─────────────────────────────────────────────────────────────
# HELPER — Emotion detection on image
# ─────────────────────────────────────────────────────────────
def detect_emotions(img: np.ndarray, anon: bool = False) -> tuple:
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cls.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    emotions_found = []

    for i, (x, y, w, h) in enumerate(faces):
        face_bgr = img[y:y+h, x:x+w]          # BGR crop for CNN
        emotion, conf = estimate_emotion(face_bgr)
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        emotions_found.append((emotion, conf))

        if anon:
            anonymize_roi(img, x, y, w, h)

        # Main rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Emotion label pill above face
        label = f"{emotion}  {conf*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x, y-lh-14), (x+lw+10, y), (0, 0, 0), -1)
        cv2.rectangle(img, (x, y-lh-14), (x+lw+10, y), color, 1)
        cv2.putText(img, label, (x+5, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Confidence bar (bottom of face)
        bar_w = int(w * conf)
        cv2.rectangle(img, (x, y+h+2), (x+w, y+h+8), (40, 40, 40), -1)
        cv2.rectangle(img, (x, y+h+2), (x+bar_w, y+h+8), color, -1)

    # HUD
    cv2.rectangle(img, (0, 0), (350, 44), (0, 0, 0), -1)
    cv2.putText(img, f"Faces: {len(faces)}  |  Emotions analyzed",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (80, 200, 255), 2)

    return img, len(faces), emotions_found

# ─────────────────────────────────────────────────────────────
# HELPER — Number plate detection
# ─────────────────────────────────────────────────────────────
def detect_plate(car_roi: np.ndarray) -> np.ndarray:
    if car_roi.size == 0:
        return car_roi
    gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h) if h > 0 else 0
            if 2.5 < aspect < 5.5 and w > 50:
                cv2.rectangle(car_roi, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(car_roi, "Plate", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                break
    return car_roi

# ─────────────────────────────────────────────────────────────
# HELPER — Centroid tracker with motion trail
# ─────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, trail_len: int = 20):
        self.next_id  = 0
        self.objects  = {}
        self.km_total = {}
        self.trails   = {}       # id → deque of centroids
        self.trail_len = trail_len

    def update(self, rects, px_per_meter: float):
        centroids = [(int(x+w/2), int(y+h/2)) for (x, y, w, h) in rects]

        if not self.objects:
            for c in centroids:
                self._register(c)
        else:
            ids   = list(self.objects.keys())
            old_c = list(self.objects.values())
            used  = set()

            for c in centroids:
                dists = [np.linalg.norm(np.array(c) - np.array(o)) for o in old_c]
                idx   = int(np.argmin(dists))
                if dists[idx] < 120 and idx not in used:
                    oid = ids[idx]
                    px_moved = np.linalg.norm(
                        np.array(c) - np.array(self.objects[oid])
                    )
                    self.km_total[oid] += (px_moved / px_per_meter) / 1000.0
                    self.objects[oid]   = c
                    self.trails[oid].append(c)
                    used.add(idx)
                else:
                    self._register(c)

        return self.objects

    def _register(self, c):
        self.objects[self.next_id]  = c
        self.km_total[self.next_id] = 0.0
        self.trails[self.next_id]   = deque(maxlen=self.trail_len)
        self.trails[self.next_id].append(c)
        self.next_id += 1

    def draw_trails(self, frame: np.ndarray) -> np.ndarray:
        for oid, trail in self.trails.items():
            trail_list = list(trail)
            for j in range(1, len(trail_list)):
                alpha = j / len(trail_list)
                color = (
                    int(50 * alpha),
                    int(150 * alpha),
                    int(255 * alpha),
                )
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trail_list[j-1], trail_list[j], color, thickness)
        return frame

# ─────────────────────────────────────────────────────────────
# HELPER — Build Plotly bar chart for detection stats
# ─────────────────────────────────────────────────────────────
def make_bar_chart(labels: list, values: list, title: str):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(
            color=values,
            colorscale=[[0, "#1e3a5f"], [0.5, "#2563eb"], [1.0, "#60a5fa"]],
            showscale=False,
        ),
        text=values,
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="#0f1629",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8", family="Inter"),
        title_font=dict(color="#60a5fa", size=14),
        xaxis=dict(gridcolor="#1e2d45", color="#64748b"),
        yaxis=dict(gridcolor="#1e2d45", color="#64748b"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
    )
    return fig

# ─────────────────────────────────────────────────────────────
# HELPER — Build crowd heatmap from centroid accumulations
# ─────────────────────────────────────────────────────────────
def build_heatmap(all_centroids: list, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (cx, cy) in all_centroids:
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(heat, (cx, cy), 30, 1.0, -1)
    # Normalize
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_uint8 = (heat * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return heatmap_color

# ─────────────────────────────────────────────────────────────
# MODE: FACE DETECTION / FACE & EYE (Image)
# ─────────────────────────────────────────────────────────────
if mode in ("face", "face_eye"):
    st.markdown(
        '<div class="info-box">📂 Upload a JPG/PNG image. '
        'Multiple faces and eyes will be detected simultaneously.</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("❌ Could not read the image. Please upload a valid JPG or PNG file.")
            st.stop()
        draw_eyes = (mode == "face_eye")
        result, fc, ec = detect_faces_eyes(img_bgr, draw_eyes, anon=anonymize_faces)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h2>{fc}</h2><p>Faces Detected</p></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h2>{ec}</h2><p>Eyes Detected</p></div>',
                        unsafe_allow_html=True)
        with col3:
            mode_txt = "Face + Eye" if draw_eyes else "Face Only"
            priv_txt = " 🔒" if anonymize_faces else ""
            st.markdown(f'<div class="metric-card"><h2>{mode_txt}</h2><p>Mode{priv_txt}</p></div>',
                        unsafe_allow_html=True)

        st.image(result, channels="BGR", use_column_width=True)

        _, buf = cv2.imencode(".jpg", result)
        st.download_button("⬇️ Download Result", buf.tobytes(),
                           file_name="detected.jpg", mime="image/jpeg")

# ─────────────────────────────────────────────────────────────
# MODE: EMOTION DETECTION (Image)
# ─────────────────────────────────────────────────────────────
elif mode == "emotion":
    st.markdown(
        '<div class="info-box">'
        '🎭 Upload a photo with faces. Each face will be analyzed for emotion: '
        '<span class="emotion-happy">Happy</span> · '
        '<span class="emotion-sad">Sad</span> · '
        '<span class="emotion-angry">Angry</span> · '
        '<span class="emotion-surprised">Surprised</span> · '
        '<span class="emotion-neutral">Neutral</span>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("❌ Could not read the image. Please upload a valid JPG or PNG file.")
            st.stop()
        result, fc, emotions_list = detect_emotions(img_bgr, anon=anonymize_faces)

        # Metrics
        emotion_names = [e for e, _ in emotions_list]
        emotion_counts = {e: emotion_names.count(e) for e in set(emotion_names)}

        cols = st.columns(max(len(emotion_counts) + 1, 3))
        cols[0].markdown(
            f'<div class="metric-card"><h2>{fc}</h2><p>Faces Analyzed</p></div>',
            unsafe_allow_html=True
        )
        for i, (emo, cnt) in enumerate(emotion_counts.items()):
            cols[i+1].markdown(
                f'<div class="metric-card"><h2>{cnt}</h2><p>{emo}</p></div>',
                unsafe_allow_html=True
            )

        st.image(result, channels="BGR", use_column_width=True)

        # Emotion breakdown chart
        if emotions_list:
            all_emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral"]
            counts = [emotion_names.count(e) for e in all_emotions]
            fig = make_bar_chart(all_emotions, counts, "Emotion Distribution")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        _, buf = cv2.imencode(".jpg", result)
        st.download_button("⬇️ Download Result", buf.tobytes(),
                           file_name="emotions_detected.jpg", mime="image/jpeg")

# ─────────────────────────────────────────────────────────────
# MODE: FULL BODY DETECTION — Haar (Video)
# ─────────────────────────────────────────────────────────────
elif mode == "body":
    st.markdown(
        '<div class="info-box">'
        '🚶 Upload an MP4 video. People will be detected, counted, '
        'and a crowd density heatmap will be generated.'
        '</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        col1, col2, col3 = st.columns(3)
        stat_people = col1.empty()
        stat_frame  = col2.empty()
        stat_max    = col3.empty()

        chart_ph = st.empty() if show_chart else None
        hmap_ph  = st.empty() if show_heatmap else None

        frame_idx   = 0
        max_people  = 0
        all_centroids = []
        detection_history = []

        st.info("⏯️ Processing video — this may take a moment …")
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            bodies = body_cls.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 100),
            )

            count = len(bodies)
            max_people = max(max_people, count)
            detection_history.append(count)

            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 230, 200), 2)
                cx, cy = x + w//2, y + h//2
                all_centroids.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, (255, 200, 0), -1)

            # FPS
            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0

            cv2.rectangle(frame, (0, 0), (360, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"People: {count}  |  FPS: {fps:.1f}",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 200), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

            stat_people.markdown(
                f'<div class="metric-card"><h2>{count}</h2><p>People in Frame</p></div>',
                unsafe_allow_html=True)
            stat_frame.markdown(
                f'<div class="metric-card"><h2>{frame_idx}</h2><p>Frames Processed</p></div>',
                unsafe_allow_html=True)
            stat_max.markdown(
                f'<div class="metric-card"><h2>{max_people}</h2><p>Peak People Count</p></div>',
                unsafe_allow_html=True)

            if show_chart and chart_ph and len(detection_history) > 1:
                window = detection_history[-30:] if len(detection_history) >= 30 else detection_history
                fig = make_bar_chart(
                    [str(i) for i in range(1, len(window)+1)],
                    window, "People Count (last 30 frames)")
                chart_ph.plotly_chart(fig, use_container_width=True)

        cap.release()

        # Final heatmap
        if show_heatmap and all_centroids and hmap_ph:
            cap2 = cv2.VideoCapture(tfile.name)
            ret2, sample_frame = cap2.read()
            cap2.release()
            if ret2:
                heatmap = build_heatmap(all_centroids, sample_frame.shape)
                alpha = 0.55
                overlay = cv2.addWeighted(sample_frame, 1 - alpha, heatmap, alpha, 0)
                st.markdown('<p class="heatmap-title">🌡️ Crowd Density Heatmap (Accumulated)</p>',
                            unsafe_allow_html=True)
                hmap_ph.image(overlay, channels="BGR", use_column_width=True)

        st.success(f"✅ Done!  Peak people detected in a single frame: **{max_people}**")

# ─────────────────────────────────────────────────────────────
# MODE: HOG PEOPLE DETECTOR (Video)  — Enhanced
# ─────────────────────────────────────────────────────────────
elif mode == "hog":
    st.markdown(
        '<div class="info-box">'
        '🕵️ <b>HOG + SVM People Detector (Enhanced)</b><br>'
        '&nbsp;✅ Person ID Tracking &nbsp;✅ Left/Right Zone Count<br>'
        '&nbsp;✅ Live Heatmap &nbsp;✅ Crowd Alert &nbsp;✅ Confidence Score<br>'
        '&nbsp;✅ Entry / Exit Counter &nbsp;✅ Motion Trails'
        '</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    crowd_alert_thresh = st.slider(
        "Crowd Alert Threshold (people)", 1, 20, 5,
        help="Alert fires when this many people appear in one frame"
    )

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap   = cv2.VideoCapture(tfile.name)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stframe = st.empty()
        col1, col2, col3, col4 = st.columns(4)
        stat_people = col1.empty();  stat_left  = col2.empty()
        stat_right  = col3.empty();  stat_max   = col4.empty()
        alert_ph    = st.empty()
        chart_ph    = st.empty() if show_chart else None

        frame_idx        = 0
        max_people       = 0
        entry_count      = 0
        exit_count       = 0
        detection_history = []
        hog_tracker      = CentroidTracker(trail_len=20)
        prev_cy          = {}
        heat_accum       = np.zeros((vid_h, vid_w), dtype=np.float32)

        st.info("⏯️ HOG enhanced detection running …")
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            # Scale down for HOG speed
            small = cv2.resize(frame, (640, 360))
            sx = vid_w / 640.0;  sy = vid_h / 360.0

            rects, weights = hog.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )

            # Scale rects back to original resolution
            scaled_rects = [
                (int(rx*sx), int(ry*sy), int(rw*sx), int(rh*sy))
                for (rx, ry, rw, rh) in rects
            ]

            count      = len(scaled_rects)
            max_people = max(max_people, count)
            detection_history.append(count)

            # ── Person tracking + entry/exit ──────────────────────────────
            objects = hog_tracker.update(scaled_rects, px_per_meter=100)
            mid_y_line = vid_h // 2
            for oid, (cx, cy) in objects.items():
                if oid in prev_cy:
                    old = prev_cy[oid]
                    if old < mid_y_line <= cy:
                        exit_count  += 1
                    elif old > mid_y_line >= cy:
                        entry_count += 1
                prev_cy[oid] = cy

            # ── Heatmap accumulation + live blend ─────────────────────────
            for (x, y, w, h) in scaled_rects:
                cx_h, cy_h = x + w//2, y + h//2
                if 0 <= cx_h < vid_w and 0 <= cy_h < vid_h:
                    cv2.circle(heat_accum, (cx_h, cy_h), 45, 1.0, -1)

            if heat_accum.max() > 0:
                h_norm  = (heat_accum / heat_accum.max() * 255).astype(np.uint8)
            else:
                h_norm  = heat_accum.astype(np.uint8)
            hmap_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_JET)
            frame      = cv2.addWeighted(frame, 0.72, hmap_color, 0.28, 0)

            # ── Zones ─────────────────────────────────────────────────────
            mid_x      = vid_w // 2
            left_count = right_count = 0
            cv2.line(frame, (mid_x, 0), (mid_x, vid_h), (255, 255, 0), 1)
            cv2.putText(frame, "LEFT",  (mid_x//4,         26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, "RIGHT", (mid_x + mid_x//4, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # ── Draw detections ───────────────────────────────────────────
            for i, (x, y, w, h) in enumerate(scaled_rects):
                cx_d     = x + w//2
                conf     = float(weights[i]) if i < len(weights) else 0.0
                conf_pct = min(int(conf * 40), 99)

                if cx_d < mid_x:
                    left_count  += 1
                    box_col = (0, 200, 255)
                else:
                    right_count += 1
                    box_col = (0, 255, 140)

                cv2.rectangle(frame, (x, y), (x+w, y+h), box_col, 2)
                lbl = f"P{i+1} | {conf_pct}%"
                (lw2, lh2), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y-lh2-8), (x+lw2+6, y), (0, 0, 0), -1)
                cv2.putText(frame, lbl, (x+3, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1)
                cv2.circle(frame, (cx_d, y+h//2), 5, (255, 210, 0), -1)

            hog_tracker.draw_trails(frame)

            # ── HUD ────────────────────────────────────────────────────────
            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0
            hud = (f"People:{count}  L:{left_count} R:{right_count}"
                   f"  In:{entry_count} Out:{exit_count}  FPS:{fps:.1f}")
            cv2.rectangle(frame, (0, vid_h-40), (vid_w, vid_h), (0, 0, 0), -1)
            cv2.putText(frame, hud, (8, vid_h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

            stat_people.markdown(
                f'<div class="metric-card"><h2>{count}</h2><p>People in Frame</p></div>',
                unsafe_allow_html=True)
            stat_left.markdown(
                f'<div class="metric-card"><h2>{left_count}</h2><p>Left Zone</p></div>',
                unsafe_allow_html=True)
            stat_right.markdown(
                f'<div class="metric-card"><h2>{right_count}</h2><p>Right Zone</p></div>',
                unsafe_allow_html=True)
            stat_max.markdown(
                f'<div class="metric-card"><h2>{max_people}</h2><p>Peak Count</p></div>',
                unsafe_allow_html=True)

            if count >= crowd_alert_thresh:
                alert_ph.error(
                    f"🚨 CROWD ALERT! {count} people detected (threshold: {crowd_alert_thresh})"
                )
            else:
                alert_ph.empty()

            if show_chart and chart_ph and len(detection_history) > 1:
                window = (detection_history[-30:]
                          if len(detection_history) >= 30 else detection_history)
                fig = make_bar_chart(
                    [str(i) for i in range(1, len(window)+1)],
                    window, "HOG People Count (last 30 frames)")
                chart_ph.plotly_chart(fig, use_container_width=True)

        cap.release()
        st.success(
            f"Done!  Peak: **{max_people}** people | "
            f"Entries: **{entry_count}** | Exits: **{exit_count}**"
        )

# ─────────────────────────────────────────────────────────────
# MODE: CAR DETECTION (Video)  — Enhanced
# ─────────────────────────────────────────────────────────────
elif mode == "car":

    uploaded = st.file_uploader("Upload Car Video", type=["mp4", "avi", "mov"])

    if uploaded:

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        LINE_Y = int(height * 0.55)

        tracker = CentroidTracker(trail_len=30)
        crossed = set()
        cross_count = 0
        prev_oid_cy = {}

        stframe = st.empty()
        frame_idx = 0

        def get_traffic_status(vehicle_count):
            if vehicle_count <= 3:
                return "Low Traffic"
            elif vehicle_count <= 7:
                return "Medium Traffic"
            else:
                return "Heavy Traffic"

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            results = yolo_model(frame)
            moving_cars = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls in [2,3,5,7] and conf > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        moving_cars.append((x1,y1,w,h))

            objects = tracker.update(moving_cars,100)

            for oid,(cx,cy) in objects.items():
                if oid in prev_oid_cy:
                    if prev_oid_cy[oid] < LINE_Y <= cy and oid not in crossed:
                        crossed.add(oid)
                        cross_count += 1
                prev_oid_cy[oid] = cy

            traffic_status = get_traffic_status(len(moving_cars))

            elapsed_minutes = (time.time()-start_time)/60
            flow_rate = cross_count/(elapsed_minutes+1e-6)

            congestion_index = (len(moving_cars)/width)*100

            cv2.line(frame,(0,LINE_Y),(width,LINE_Y),(0,0,255),3)

            for (x,y,w,h) in moving_cars:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

            tracker.draw_trails(frame)

            current_time = datetime.now().strftime("%H:%M:%S")

            new_row = {
                "Time": current_time,
                "Frame_Number": frame_idx,
                "Vehicle_Count": len(moving_cars),
                "Crossed_Count": cross_count,
                "Traffic_Status": traffic_status,
                "Flow_Rate": flow_rate,
                "Congestion_Index": congestion_index
            }

            log_data.loc[len(log_data)] = new_row

            cursor.execute("""
            INSERT INTO vehicle_log (time, frame, vehicle_count, crossed_count)
            VALUES (?, ?, ?, ?)
            """,(current_time,frame_idx,len(moving_cars),cross_count))
            conn.commit()

            cv2.putText(frame,
            f"Cars:{len(moving_cars)}  Crossed:{cross_count}  {traffic_status}",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,255,255),2)

            stframe.image(frame,channels="BGR")

        cap.release()

        log_data.to_csv("vehicle_traffic_log.csv",index=False)

        st.subheader("📊 Traffic Analytics Dashboard")

        df = pd.read_csv("vehicle_traffic_log.csv")

        st.dataframe(df.tail())

        avg_traffic = df["Vehicle_Count"].mean()
        peak_traffic = df["Vehicle_Count"].max()

        st.metric("Average Vehicles", int(avg_traffic))
        st.metric("Peak Traffic", int(peak_traffic))
        st.metric("Vehicles Per Minute", int(flow_rate))

        df["Time"]=pd.to_datetime(df["Time"])
        df["Hour"]=df["Time"].dt.hour

        peak_hour = df.groupby("Hour")["Vehicle_Count"].mean().idxmax()
        st.metric("Peak Traffic Hour", peak_hour)

        model = LinearRegression()
        X=df[['Frame_Number']]
        y=df['Vehicle_Count']
        model.fit(X,y)

        future_frames=np.array([[frame_idx+30]])
        future_traffic=model.predict(future_frames)
        st.metric("Predicted Vehicles (Next 10 sec)",int(future_traffic[0]))

        if len(moving_cars)>8:
            st.error("⚠️ High Accident Risk Zone")

        if flow_rate>15:
            st.warning("🚦 Suggestion: Increase Green Signal Time")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Frame_Number"],
            y=df["Vehicle_Count"],
            mode='lines',
            name='Traffic Flow'
        ))
        st.plotly_chart(fig)

        fig2=go.Figure()
        fig2.add_trace(go.Bar(
            x=df["Time"],
            y=df["Vehicle_Count"]
        ))
        fig2.update_layout(title="Traffic Density Over Time")
        st.plotly_chart(fig2)

        st.success("Vehicle Traffic Analytics Generated Successfully!")
            

# ─────────────────────────────────────────────────────────────
# MODE: LIVE WEBCAM — Multi-face + eye + snapshot
# ─────────────────────────────────────────────────────────────
elif mode == "webcam":
    st.markdown(
        '<div class="info-box">'
        '🎥 Click <b>START</b> to open your webcam. All faces and eyes are detected in real time.<br>'
        '📸 Use the <b>Snapshot</b> button to capture and download a frame.'
        '</div>',
        unsafe_allow_html=True
    )

    # Shared snapshot queue
    snapshot_queue: queue.Queue = queue.Queue(maxsize=1)

    class MultiDetectionTransformer(VideoTransformerBase):
        def __init__(self):
            self._do_snap = False

        def recv_snap(self):
            self._do_snap = True

        def transform(self, frame):
            img  = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cls.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            total_eyes = 0

            for i, (x, y, w, h) in enumerate(faces):
                if anonymize_faces:
                    anonymize_roi(img, x, y, w, h)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 255), 2)
                    cv2.putText(img, f"Anon {i+1}", (x, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
                else:
                    # Emotion label via CNN — pass BGR face crop
                    face_bgr = img[y:y+h, x:x+w]
                    emotion, conf = estimate_emotion(face_bgr)
                    emo_color = EMOTION_COLORS.get(emotion, (180, 180, 180))

                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 80, 80), 2)
                    label = f"Face {i+1} | {emotion}"
                    cv2.putText(img, label, (x, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, emo_color, 2)

                    # Eye detection
                    eye_y_end = int(y + h * 0.60)
                    roi_gray  = gray[y:eye_y_end, x:x+w]
                    roi_color = img [y:eye_y_end, x:x+w]

                    eyes = eye_cls.detectMultiScale(
                        roi_gray, scaleFactor=1.08, minNeighbors=6,
                        minSize=(10, 10), maxSize=(int(w*0.45), int(h*0.35)),
                    )
                    valid = sorted(eyes, key=lambda e: e[0])[:2]
                    total_eyes += len(valid)
                    for (ex, ey, ew, eh) in valid:
                        cv2.rectangle(roi_color, (ex, ey),
                                      (ex+ew, ey+eh), (0, 230, 100), 2)

            # HUD
            hud = f"Faces: {len(faces)}  Eyes: {total_eyes}"
            cv2.rectangle(img, (0, 0), (300, 44), (0, 0, 0), -1)
            cv2.putText(img, hud, (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2)

            # Snapshot: push to queue if requested
            if self._do_snap:
                self._do_snap = False
                try:
                    snapshot_queue.put_nowait(img.copy())
                except queue.Full:
                    pass

            return img

    if WEBRTC_AVAILABLE:
        ctx = webrtc_streamer(
            key="multi-detect-v2",
            video_transformer_factory=MultiDetectionTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )

        col_snap, col_dl = st.columns([1, 3])
        with col_snap:
            if st.button("📸 Snapshot") and ctx.video_transformer:
                ctx.video_transformer.recv_snap()
                st.info("Snapshot requested — check below in ~2 sec")

        # Show latest snapshot
        try:
            snap = snapshot_queue.get_nowait()
            st.markdown('<div class="snap-card">📸 <b>Latest Snapshot</b></div>',
                        unsafe_allow_html=True)
            st.image(snap, channels="BGR", use_column_width=True)
            _, buf = cv2.imencode(".png", snap)
            st.download_button("⬇️ Download Snapshot", buf.tobytes(),
                               file_name="snapshot.png", mime="image/png")
        except queue.Empty:
            pass
    else:
        st.error("⚠️ streamlit-webrtc is not installed. "
                 "Run: `pip install streamlit-webrtc` to enable webcam mode.")

    st.markdown("""
    <div class="info-box">
    💡 <b>Tips for best results:</b><br>
    • Ensure good, even front lighting on your face<br>
    • Look directly at the camera<br>
    • Works with multiple people simultaneously<br>
    • Emotion labels appear next to each face in real time
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:0.78rem;">'
    '🚀 Smart Object Detection v2.0 &nbsp;|&nbsp; OpenCV · HOG+SVM · Streamlit'
    '</p>',
    unsafe_allow_html=True
)
