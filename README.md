# 🎯 VisionLock-AI

A real-time AI-based Smart Surveillance System built using **YOLOv8**, **ByteTrack**, **OpenCV**, and **Streamlit**. The system detects and tracks multiple objects, allows users to lock onto a target, and displays real-time tracking information.

---

## 🚀 Features

- 🎥 Real-time object detection using YOLOv8
- 🔄 Multi-object tracking using ByteTrack
- 🎯 Target locking based on Tracking ID
- 📍 Real-time X & Y offset calculation
- 🔍 Target zoom feature
- 📊 Confidence score display
- 🖥️ Streamlit web application
- 📹 Video upload and processing

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- Streamlit
- NumPy

---

## 📂 Project Structure

```
VisionLock-AI/
│── app.py
│── main.py
│── input_video.mp4
│── requirements.txt
│── README.md
│── .gitignore
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

## ▶️ Run Desktop Version

```bash
python main.py
```

---

## ▶️ Run Streamlit Version

```bash
streamlit run app.py
```

---



---

## 🎯 Future Improvements

- Mouse-click target selection in Streamlit
- Object re-identification
- Live webcam support in Streamlit
- Detection history logging
- MySQL integration
- Dashboard analytics

---

## 👩‍💻 Author

**Mansi Nirbhavane**
