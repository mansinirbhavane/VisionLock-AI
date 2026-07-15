# 🛡️ VisionLock-AI

## Multi-Modal Intelligent Surveillance Platform

VisionLock-AI is a real-time AI-powered surveillance platform that combines **RGB object detection**, **thermal human detection**, **multi-object tracking**, and **target locking** into a single intelligent system.

The platform leverages **Ultralytics YOLO**, **ByteTrack**, **OpenCV**, and **Streamlit** to perform real-time analysis on both RGB and thermal imagery, making it suitable for smart surveillance, security monitoring, and AI vision applications.

---

## 🚀 Features

### 🟢 RGB Vision

- Real-time object detection
- Multi-object tracking using ByteTrack
- Target locking using Tracking ID
- Live target offset calculation
- Zoomed target view
- Confidence score visualization

### 🔥 Thermal Vision

- Thermal human detection
- Thermal video surveillance
- Real-time human tracking
- Support for pretrained thermal YOLO model
- Thermal image inference
- Thermal video inference

### 💻 Dashboard

- Streamlit-based user interface
- Image & video upload
- Multiple AI modules
- Interactive controls
- Real-time visualization

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- Streamlit
- NumPy

---

## 🧠 AI Modules

| Module | Description |
|---------|-------------|
| RGB Object Detection | Detects multiple objects in RGB videos |
| RGB Target Tracking | Tracks objects using ByteTrack |
| Target Locking | Locks onto a selected target and calculates offset |
| Thermal Human Detection | Detects humans from thermal images |
| Thermal Video Tracking | Detects and tracks humans in thermal videos |

---

## 📂 Project Structure

```text
VisionLock-AI/
│
├── app.py
├── rgb_target_tracking.py
├── rgb_surveillance.py
├── thermal_detection.py
├── thermal_tracking.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── yolov8n.pt
│   ├── thermal_yolo.pt
│   ├── best.pt
│   └── best1.pt
│
├── sample_inputs/
│
└── sample_outputs/
```

---

## ⚙️ Installation

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

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📸 Demo

The repository includes demonstration videos showcasing:

- RGB Object Detection
- Target Locking
- Thermal Human Detection
- Thermal Video Tracking

---

## 📈 Future Improvements

- Face Recognition
- Automatic Threat Detection
- Fire & Smoke Detection
- License Plate Recognition
- Drone Camera Support
- Person Re-identification
- Cloud Deployment
- SQL Database Integration
- Analytics Dashboard
- Event Logging & Report Generation

---

## 🎯 Applications

- Smart City Surveillance
- Railway Safety Monitoring
- Border Security
- Industrial Safety
- Military Surveillance
- Wildlife Monitoring
- Critical Infrastructure Protection

---

## 👩‍💻 Author

**Mansi Nirbhavane**

Computer Engineering | AI/ML | Computer Vision | Deep Learning

---

⭐ If you found this project useful, consider giving it a Star!