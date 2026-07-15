from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from collections import Counter

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
model = YOLO("yolov8n.pt")
# --------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------
os.makedirs("captures", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("logs", exist_ok=True)
# --------------------------------
# OPEN WEBCAM
# --------------------------------
cap = cv2.VideoCapture(0)

# --------------------------------
# FPS VARIABLES
# --------------------------------
prev_time = time.time()
fps = 0
# --------------------------------
# RECORDING VARIABLES
# --------------------------------
recording = False
video_writer = None
# --------------------------------
# GLOBAL VARIABLES
# --------------------------------
locked_id = -1
current_boxes = []
# --------------------------------
# DETECTION LOG FILE
# --------------------------------
log_file = "logs/detection_log.csv"

if not os.path.exists(log_file):

    with open(log_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Timestamp",
            "Tracking ID",
            "Object",
            "Confidence"
        ])
# --------------------------------
# IMPORTANT OBJECTS
# --------------------------------
important_objects = [
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle"
]

# --------------------------------
# MOUSE CLICK FUNCTION
# --------------------------------
def mouse_callback(event, x, y, flags, param):

    global locked_id
    global current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:

        target_found = False

        # Search all tracked objects
        for x1, y1, x2, y2, track_id in current_boxes:

            if x1 <= x <= x2 and y1 <= y <= y2:

                locked_id = track_id
                target_found = True

                print("=" * 50)
                print(f"TARGET LOCKED")
                print(f"Tracking ID : {locked_id}")
                print("=" * 50)

                break

        # Clicked outside every object
        if not target_found:

            locked_id = -1
            print("TARGET UNLOCKED")

# --------------------------------
# CREATE DISPLAY WINDOW
# --------------------------------
WINDOW_NAME = "VisionLock-AI : RGB Live Surveillance"

cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

# --------------------------------
# MAIN LOOP
# --------------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------
    # FPS CALCULATION
    # --------------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # --------------------------------
    # RESIZE FRAME
    # --------------------------------
    frame = cv2.resize(frame, (1000, 700))

    # DISPLAY FRAME
    display_frame = frame.copy()

    # --------------------------------
    # FRAME CENTER
    # --------------------------------
    frame_center_x = display_frame.shape[1] // 2
    frame_center_y = display_frame.shape[0] // 2

    # --------------------------------
    # CENTER CROSSHAIR
    # --------------------------------
    cv2.circle(
        display_frame,
        (frame_center_x, frame_center_y),
        8,
        (0, 0, 255),
        -1
    )

    cv2.line(
        display_frame,
        (frame_center_x - 20, frame_center_y),
        (frame_center_x + 20, frame_center_y),
        (0, 0, 255),
        2
    )

    cv2.line(
        display_frame,
        (frame_center_x, frame_center_y - 20),
        (frame_center_x, frame_center_y + 20),
        (0, 0, 255),
        2
    )

    # --------------------------------
    # RUN TRACKING
    # --------------------------------
    results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.18
        )


    # --------------------------------
    # OBJECT COUNTER
    # --------------------------------
    boxes = results[0].boxes
    object_counter = Counter()

    # CLEAR BOXES
    current_boxes.clear()

    # --------------------------------
    # PROCESS DETECTIONS
    # --------------------------------
    if boxes is not None:

            for box in boxes:

                if box.id is None:
                    continue

                # TRACK ID
                track_id = int(box.id[0])

                # CLASS INFO
                class_id = int(box.cls[0])

                class_name = model.names[class_id]
                object_counter[class_name] += 1

                confidence_score = float(box.conf[0])
                # --------------------------------
                # SAVE DETECTION LOG
                # --------------------------------
                with open(log_file, "a", newline="") as f:

                 writer = csv.writer(f)

                 writer.writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    track_id,
                    class_name,
                    round(confidence_score, 2)
                ])
                # BOUNDING BOX
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # SAVE FOR CLICKING
                current_boxes.append(
                    (x1, y1, x2, y2, track_id)
                )

                # OBJECT CENTER
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                # --------------------------------
                # LOCKED TARGET
                # --------------------------------
                if track_id == locked_id:

                    # RED TARGET BOX
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        4
                    )

                    # LABEL
                    cv2.putText(
                        display_frame,
                        f"LOCKED: {class_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    # TARGET STATUS
                    cv2.putText(
                        display_frame,
                        "TARGET LOCKED",
                        (350, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    # --------------------------------
                    # OFFSET CALCULATIONS
                    # --------------------------------
                    offset_x = obj_center_x - frame_center_x
                    offset_y = obj_center_y - frame_center_y

                    # TARGET LINE
                    cv2.line(
                        display_frame,
                        (frame_center_x, frame_center_y),
                        (obj_center_x, obj_center_y),
                        (255, 0, 0),
                        2
                    )

                    # VISUAL RECENTERING GUIDE
                    cv2.arrowedLine(
                        display_frame,
                        (frame_center_x, frame_center_y),
                        (obj_center_x, obj_center_y),
                        (0, 255, 255),
                        3
                    )

                    # OFFSETS
                    cv2.putText(
                        display_frame,
                        f"Offset X: {offset_x}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"Offset Y: {offset_y}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # --------------------------------
                    # TARGET INFO PANEL
                    # --------------------------------
                    overlay = display_frame.copy()

                    cv2.rectangle(
                        overlay,
                        (20, 110),
                        (320, 250),
                        (25, 25, 25),
                        -1
                    )

                    display_frame = cv2.addWeighted(
                        overlay,
                        0.65,
                        display_frame,
                        0.35,
                        0
                    )

                    cv2.rectangle(
                        display_frame,
                        (20, 110),
                        (320, 250),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        "TARGET INFORMATION",
                        (35, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"Class: {class_name}",
                        (35, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"Tracking ID: {track_id}",
                        (35, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"Confidence: {confidence_score:.2f}",
                        (35, 235),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # --------------------------------
                    # TARGET ZOOM PANEL
                    # --------------------------------
                    target_crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)
                    ]

                    if target_crop.size > 0:

                        zoom_target = cv2.resize(
                            target_crop,
                            (220, 220)
                        )

                        display_frame[20:240, 740:960] = zoom_target

                        cv2.rectangle(
                            display_frame,
                            (740, 20),
                            (960, 240),
                            (0, 0, 255),
                            3
                        )

                        cv2.putText(
                            display_frame,
                            "TARGET ZOOM",
                            (770, 270),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                else:

                    # IMPORTANT OBJECTS
                    if class_name in important_objects:

                        color = (0, 255, 255)

                    else:

                        color = (0, 255, 0)

                    # NORMAL OBJECT BOX
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"{class_name} ID:{track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

    # --------------------------------
    # AI DASHBOARD
    # --------------------------------
    overlay = display_frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (290, 150),
        (30, 30, 30),
        -1
    )

    display_frame = cv2.addWeighted(
        overlay,
        0.6,
        display_frame,
        0.4,
        0
    )

    cv2.rectangle(
        display_frame,
        (10, 10),
        (290, 150),
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "VisionLock-AI",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"FPS : {int(fps)}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Objects : {sum(object_counter.values())}",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    status = "YES" if locked_id != -1 else "NO"

    cv2.putText(
        display_frame,
        f"Locked : {status}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # RECORD VIDEO
    # --------------------------------
    if recording:

        video_writer.write(display_frame)

    # --------------------------------
    # SHOW OUTPUT
    # --------------------------------
    cv2.imshow(
    WINDOW_NAME,
    display_frame
)

    # --------------------------------
    # KEYBOARD CONTROLS
    # --------------------------------
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
       break

    # Screenshot
    elif key == ord("s"):

        filename = f"captures/capture_{int(time.time())}.jpg"

        cv2.imwrite(
            filename,
            display_frame
        )

        print(f"Screenshot Saved -> {filename}")

    # Record
    elif key == ord("r"):

        if not recording:

            filename = f"recordings/output_{int(time.time())}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                20,
                (
                    display_frame.shape[1],
                    display_frame.shape[0]
                )
            )

            recording = True

            print("Recording Started")

        else:

            recording = False

            if video_writer is not None:
                video_writer.release()

            print("Recording Stopped")

    # --------------------------------
    # RELEASE
    # --------------------------------
cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()