# 🛡️ VisionLock-AI

> **Multi-Modal Intelligent Surveillance Platform using Computer Vision and Deep Learning**

VisionLock-AI is a real-time AI-powered surveillance system that combines **RGB** and **Thermal Vision** for intelligent object detection, human detection, multi-object tracking, and target locking. The platform provides both a **desktop surveillance application** and an **interactive Streamlit dashboard** for monitoring recorded videos and live surveillance.

---

## 🚀 Key Features

### 🎥 RGB Live Surveillance
- Live webcam surveillance
- Real-time object detection using YOLOv8
- Multi-object tracking using ByteTrack
- Click-to-lock target selection
- Target zoom panel
- Target information panel
- Object counter
- FPS monitoring
- Screenshot capture
- Video recording
- Detection logging (CSV)

### 📹 RGB Video Tracking
- Upload RGB videos
- Detect and track multiple objects
- Target locking using Tracking ID
- Zoom selected target
- Offset calculation from frame center

### 🔥 Thermal Human Detection
- Human detection on thermal images
- Optimized YOLO model for thermal imagery
- Confidence score visualization

### 🌡️ Thermal Video Tracking
- Human detection in thermal videos
- Multi-object tracking using ByteTrack
- Smooth real-time tracking

---

# 🧠 Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- Streamlit
- NumPy

---

# 📂 Project Structure

```
VisionLock-AI/
│
├── app.py
├── rgb_surveillance.py
├── rgb_target_tracking.py
├── thermal_detection.py
├── thermal_tracking.py
├── sample_inputs/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/mansinirbhavane/VisionLock-AI.git
cd VisionLock-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 🎯 Dashboard Modules

The Streamlit dashboard provides four AI-powered surveillance modules:

- 🏠 Home
- 🎥 RGB Video Tracking
- 🔥 Thermal Human Detection
- 🌡️ Thermal Video Tracking
- 📡 Launch RGB Live Surveillance

---

# 📁 Sample Inputs

The repository includes sample files for quick testing.

```
sample_inputs/
│
├── input_video.mp4
└── thermal_video.mp4
```

---

# 💻 Desktop Surveillance Features

The Live Surveillance module includes:

- Multi-object tracking
- Click-to-lock target
- Target information panel
- Target zoom panel
- Real-time offset calculation
- Object counting
- FPS display
- CSV detection logging
- Screenshot capture
- Video recording

---

# 🎯 Applications

VisionLock-AI can be used in:

- Smart City Surveillance
- Border Security
- Military Reconnaissance
- Industrial Safety Monitoring
- Critical Infrastructure Protection
- Search & Rescue Operations
- Low-Light and Night Surveillance

---

# 🚀 Future Improvements

- Drone camera integration
- PTZ camera support
- Face recognition
- Object re-identification
- Cloud deployment
- Mobile application
- Multi-camera surveillance

---

# 👩‍💻 Author

**Mansi Santosh Nirbhavane**

AI/ML Engineer | Computer Vision | Deep Learning | Python

GitHub:
https://github.com/mansinirbhavane

---

⭐ If you found this project useful, consider giving it a star!
