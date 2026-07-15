import streamlit as st
from ultralytics import YOLO
import cv2

# --------------------------------
# LOAD MODEL
# --------------------------------
model = YOLO("models/thermal_yolo.pt")


# --------------------------------
# THERMAL DETECTION MODULE
# --------------------------------
def run():

    st.title("🔥 Thermal Human Detection")

    st.write(
        "Upload a thermal image to detect humans using the custom YOLO model."
    )

    uploaded_file = st.file_uploader(
        "Choose Thermal Image",
        type=["jpg", "jpeg", "png"],
        key="thermal_detection"
    )

    if uploaded_file is not None:

        image_path = "thermal_input.jpg"

        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = cv2.imread(image_path)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb,
            caption="Uploaded Thermal Image",
            use_container_width=True
        )

        results = model.predict(
            source=image_path,
            conf=0.25
        )

        output = results[0].plot()

        output = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            output,
            caption="Detection Output",
            use_container_width=True
        )

        st.success("✅ Thermal Human Detection Completed")


# --------------------------------
# RUN DIRECTLY
# --------------------------------
if __name__ == "__main__":
    run()