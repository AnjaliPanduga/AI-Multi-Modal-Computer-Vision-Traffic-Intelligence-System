# 🚀 AI-Based Multi-Modal Computer Vision & Traffic Intelligence System

## 📌 Project Overview

This project is an AI-powered multi-modal computer vision platform that performs real-time analytics on images, videos, and live webcam streams.

The system integrates object detection, facial analysis, traffic monitoring, crowd analytics, and predictive modeling into a unified interactive dashboard built with Streamlit.

---

## 🎯 Supported Modalities

- 🖼 Image Processing
- 🎥 Video Analytics
- 📷 Live Webcam Detection

---

## 🛠 Technology Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- HOG + SVM (Pedestrian Detection)
- Haar Cascades (Face, Eye, Smile Detection)
- Streamlit (Interactive Dashboard)
- SQLite (Data Logging)
- Scikit-Learn (Linear Regression Prediction)
- Plotly (Data Visualization)
- NumPy & Pandas

---

## 🔥 Core Features

### 👤 Face & Emotion Intelligence
- Multi-face detection
- Eye detection
- Smile detection
- Rule-based multi-signal emotion estimation
- Face anonymization (privacy mode)
- Real-time webcam emotion labeling

---

### 🚗 Vehicle Detection & Traffic Analytics
- YOLOv8-based vehicle detection
- Centroid-based vehicle tracking
- Line-crossing vehicle counter
- Traffic flow rate calculation
- Congestion index estimation
- Traffic status classification (Low / Medium / Heavy)
- Peak traffic hour detection
- Short-term traffic prediction using Linear Regression
- Accident risk alert logic
- SQLite-based traffic logging

---

### 🚶 Crowd Monitoring System
- HOG + SVM pedestrian detection
- Person ID tracking
- Entry / Exit counting
- Zone-based counting (Left / Right)
- Live crowd alert threshold
- Motion trail visualization
- Crowd density heatmap generation

---

### 📊 Smart Analytics Dashboard
- Real-time vehicle count visualization
- Peak traffic metrics
- Vehicles per minute estimation
- Hourly traffic analysis
- Predictive analytics for future traffic
- Interactive Plotly charts

---

## 📂 Project Structure

```
AI-Multi-Modal-Computer-Vision-Traffic-Intelligence-System/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Run Locally

1️⃣ Clone the repository:
```
git clone https://github.com/AnjaliPanduga/AI-Multi-Modal-Computer-Vision-Traffic-Intelligence-System
```

2️⃣ Install dependencies:
```
pip install -r requirements.txt
```

3️⃣ Run the application:
```
streamlit run app.py
```


















