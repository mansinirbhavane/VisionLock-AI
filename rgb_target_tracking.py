import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
model = YOLO("models/yolov8n.pt")


# --------------------------------
# RGB TARGET TRACKING MODULE
# --------------------------------
def run():

    # --------------------------------
    # PAGE TITLE
    # --------------------------------
    st.title("🎯 RGB Video Target Tracking")

    st.write(
        "Upload a video for object tracking and target locking."
    )

    # --------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------
    st.sidebar.header("Controls")

    confidence = st.sidebar.slider(
        "Confidence Threshold",
        0.1,
        1.0,
        0.25
    )

    zoom = st.sidebar.slider(
        "Zoom Level",
        1,
        3,
        1
    )

    target_id_input = st.sidebar.number_input(
        "Target ID To Lock",
        min_value=0,
        step=1
    )

    # --------------------------------
    # VIDEO UPLOAD
    # --------------------------------
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4"],
        key="rgb_tracking"
    )

    # --------------------------------
    # PROCESS VIDEO
    # --------------------------------
    if uploaded_video is not None:

        with open("input_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("input_video.mp4")

        frame_placeholder = st.empty()

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % 3 != 0:
                continue

            frame = cv2.resize(frame, (800, 500))

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=960
            )

            boxes = results[0].boxes

            cv2.circle(
                frame,
                (frame_center_x, frame_center_y),
                8,
                (0, 0, 255),
                -1
            )

            if boxes is not None:

                for box in boxes:

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2

                    if track_id == target_id_input:

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            3
                        )

                        cv2.putText(
                            frame,
                            f"LOCKED ID: {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        cv2.line(
                            frame,
                            (frame_center_x, frame_center_y),
                            (obj_center_x, obj_center_y),
                            (255, 0, 0),
                            2
                        )

                        offset_x = obj_center_x - frame_center_x
                        offset_y = obj_center_y - frame_center_y

                        cv2.putText(
                            frame,
                            f"Offset X: {offset_x}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Offset Y: {offset_y}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

                        if zoom > 1:

                            crop = frame[
                                max(0, y1):min(frame.shape[0], y2),
                                max(0, x1):min(frame.shape[1], x2)
                            ]

                            if crop.size > 0:

                                crop = cv2.resize(
                                    crop,
                                    None,
                                    fx=zoom,
                                    fy=zoom
                                )

                                h, w = crop.shape[:2]

                                if h < frame.shape[0] and w < frame.shape[1]:
                                    frame[0:h, 0:w] = crop

                    else:

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"ID: {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

        st.success("✅ Processing Completed")


# --------------------------------
# RUN DIRECTLY
# --------------------------------
if __name__ == "__main__":
    run()