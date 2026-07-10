from ultralytics import YOLO
import cv2
import numpy as np

# --------------------------------
# LOAD YOLO MODEL
# --------------------------------
model = YOLO("yolov8n.pt")

# --------------------------------
# OPEN WEBCAM
# --------------------------------
cap = cv2.VideoCapture(0)

# --------------------------------
# GLOBAL VARIABLES
# --------------------------------
locked_id = -1
current_boxes = []

# --------------------------------
# IMPORTANT OBJECTS
# --------------------------------
important_objects = [
    "cell phone",
    "bottle",
    "cup",
    "laptop",
    "remote"
]

# --------------------------------
# MOUSE CLICK FUNCTION
# --------------------------------
def mouse_callback(event, x, y, flags, param):

    global locked_id

    if event == cv2.EVENT_LBUTTONDOWN:

        for item in current_boxes:

            x1, y1, x2, y2, track_id = item

            if x >= x1 and x <= x2 and y >= y1 and y <= y2:

                locked_id = track_id

                print(f"TARGET LOCKED ID: {locked_id}")

# --------------------------------
# CREATE WINDOW
# --------------------------------
cv2.namedWindow("AI Smart Surveillance System")

cv2.setMouseCallback(
    "AI Smart Surveillance System",
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

    boxes = results[0].boxes

    # CLEAR BOXES
    current_boxes = []

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

            confidence_score = float(box.conf[0])

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
    # SHOW OUTPUT
    # --------------------------------
    cv2.imshow(
        "AI Smart Surveillance System",
        display_frame
    )

    # --------------------------------
    # EXIT
    # --------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------
# RELEASE
# --------------------------------
cap.release()

cv2.destroyAllWindows()