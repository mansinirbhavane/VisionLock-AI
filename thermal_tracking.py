import streamlit as st
from ultralytics import YOLO
import cv2

# --------------------------------
# LOAD MODEL
# --------------------------------
model = YOLO("models/thermal_yolo.pt")


# --------------------------------
# THERMAL TRACKING MODULE
# --------------------------------
def run():

    st.title("🔥 Thermal Object Tracking")

    st.write(
        "Upload a thermal video to perform human detection and tracking."
    )

    uploaded_video = st.file_uploader(
        "Choose Thermal Video",
        type=["mp4"],
        key="thermal_tracking"
    )

    if uploaded_video is not None:

        video_path = "thermal_video.mp4"

        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(video_path)

        frame_placeholder = st.empty()

        st.success("✅ Video Loaded Successfully")

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # --------------------------------
            # KEEP ORIGINAL RESOLUTION
            # --------------------------------

            # --------------------------------
            # RUN TRACKING
            # --------------------------------
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.10,
                iou=0.5,
                imgsz=1280,
                verbose=False
            )

            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

        st.success("✅ Thermal Tracking Completed")


# --------------------------------
# RUN DIRECTLY
# --------------------------------
if __name__ == "__main__":
    run()