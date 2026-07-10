import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
model = YOLO("yolov8n.pt")

# --------------------------------
# PAGE TITLE
# --------------------------------
st.title("AI Smart Target Locking System")

st.write(
    "Upload a video for object tracking and target locking"
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
    type=["mp4"]
)

# --------------------------------
# PROCESS VIDEO
# --------------------------------
if uploaded_video is not None:

    # Save uploaded video
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    # Open video
    cap = cv2.VideoCapture("input_video.mp4")

    # Streamlit frame placeholder
    frame_placeholder = st.empty()

    # Frame skipping for speed
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------
        # SKIP FRAMES FOR PERFORMANCE
        # --------------------------------
        frame_count += 1

        if frame_count % 3 != 0:
            continue

        # --------------------------------
        # RESIZE SMALLER FOR SPEED
        # --------------------------------
        frame = cv2.resize(frame, (800, 500))

        # Frame center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # --------------------------------
        # RUN BYTE TRACKING
        # --------------------------------
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            imgsz=960
        )

        boxes = results[0].boxes

        # --------------------------------
        # DRAW CENTER POINT
        # --------------------------------
        cv2.circle(
            frame,
            (frame_center_x, frame_center_y),
            8,
            (0, 0, 255),
            -1
        )

        # --------------------------------
        # PROCESS OBJECTS
        # --------------------------------
        if boxes is not None:

            for box in boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Object center
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                # --------------------------------
                # LOCKED TARGET
                # --------------------------------
                if track_id == target_id_input:

                    # Red locked target box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

                    # Locked label
                    cv2.putText(
                        frame,
                        f"LOCKED ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    # Draw center line
                    cv2.line(
                        frame,
                        (frame_center_x, frame_center_y),
                        (obj_center_x, obj_center_y),
                        (255, 0, 0),
                        2
                    )

                    # --------------------------------
                    # OFFSET CALCULATION
                    # --------------------------------
                    offset_x = obj_center_x - frame_center_x
                    offset_y = obj_center_y - frame_center_y

                    # Display offsets
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

                    # --------------------------------
                    # ZOOM FEATURE
                    # --------------------------------
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

                    # Normal tracked objects
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

        # --------------------------------
        # CONVERT BGR → RGB
        # --------------------------------
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------
        # DISPLAY FRAME
        # --------------------------------
        frame_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    st.success("Processing Completed")