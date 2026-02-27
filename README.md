# 🚀 AI-Based Multi-Modal Computer Vision & Traffic Intelligence System

## 📌 Project Overview

The AI-Based Multi-Modal Computer Vision & Traffic Intelligence System is an integrated real-time computer vision and analytics platform designed to process multiple input modalities including static images, recorded videos, and live webcam streams.

The system combines classical computer vision techniques with deep learning-based object detection models to perform:

• Face and emotion detection  
• Pedestrian tracking and crowd analytics  
• Vehicle detection and traffic monitoring  
• Entry and exit counting  
• Congestion estimation and traffic flow measurement  
• Predictive traffic analysis using machine learning  

For vehicle detection, the system uses YOLOv8 to accurately identify cars, buses, and trucks from video frames. A centroid-based tracking algorithm assigns unique IDs to vehicles and monitors line-crossing events to compute traffic flow rates.

Pedestrian detection is implemented using HOG + SVM, enabling real-time crowd monitoring with zone-based analytics, motion trails, and heatmap generation for density estimation.

Emotion detection is implemented using multi-signal facial feature analysis, incorporating smile detection, edge density, brightness metrics, and region-based intensity analysis to classify emotions such as Happy, Sad, Angry, Surprised, and Neutral.

All detection results are logged into a structured SQLite database, enabling traffic analysis, peak hour identification, congestion indexing, and statistical reporting. The system also integrates a Linear Regression model to predict short-term traffic trends.

An interactive Streamlit dashboard visualizes real-time metrics, historical data trends, traffic density charts, and predictive analytics, making the platform suitable for smart city monitoring and intelligent surveillance applications.

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
git clone https://github.com/AnjaliPanduga/AI-Multi-Modal-Computer-Vision-Traffic-Intelligence-System.git
cd AI-Multi-Modal-Computer-Vision-Traffic-Intelligence-System

```

2️⃣ Install dependencies:
```
pip install -r requirements.txt
```

3️⃣ Run the application:
```
streamlit run app.py
```

---

## 👩‍💻 Author

Anjali Panduga  

Aspiring Data Analyst | AI & Computer Vision Enthusiast

📧 Email: pandugaanjali2003@gmail.com

🔗 GitHub: https://github.com/AnjaliPanduga

---















