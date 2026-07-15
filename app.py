import streamlit as st
import os

from thermal_detection import run as thermal_detection
from thermal_tracking import run as thermal_tracking
from rgb_target_tracking import run as rgb_tracking

st.set_page_config(
    page_title="VisionLock-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------

st.title("🛡️ VisionLock-AI")
st.subheader("Multi-Modal Intelligent Surveillance Platform")

st.markdown("---")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

option = st.sidebar.radio(
    "Choose Module",
    [
        "🏠 Home",
        "📷 RGB Live Surveillance",
        "🎯 RGB Video Analysis",
        "🔥 Thermal Human Detection",
        "🔥 Thermal Video Tracking"
    ]
)

# ----------------------------------------------------
# HOME
# ----------------------------------------------------

if option == "🏠 Home":

    st.markdown("""
Welcome to **VisionLock-AI**, an AI-powered surveillance platform designed to detect, monitor, and track objects and humans using **RGB and Thermal Vision**.

Select a module from the sidebar to begin.
""")

    st.markdown("---")

    st.header("📤 Choose the Appropriate Input")

    st.info("""
📷 **RGB Live Surveillance** → Uses your system webcam.

🎯 **RGB Video Analysis** → Upload an RGB Video (.mp4)

🔥 **Thermal Human Detection** → Upload a Thermal Image (.jpg, .jpeg, .png)

🔥 **Thermal Video Tracking** → Upload a Thermal Video (.mp4)
""")

    st.markdown("---")

    st.header("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("Real-time Object Detection")
        st.success("Multi-Object Tracking")
        st.success("Intelligent Target Locking")

    with col2:

        st.success("RGB & Thermal Vision Support")
        st.success("Live Surveillance")
        st.success("Video Analysis")

    st.markdown("---")

    st.header("💡 Before You Begin")

    st.warning("""
• Use clear and high-quality images or videos for better results.

• Ensure the uploaded file matches the selected module.

• Allow webcam access when using RGB Live Surveillance.

• Upload only thermal images/videos for thermal modules.
""")

    st.markdown("---")

    st.header("🌍 Applications")

    st.markdown("""
- 🚆 Railway Safety & Track Monitoring

- 🏭 Industrial & Factory Surveillance

- 🌃 Smart City Surveillance

- 🛡️ Border & Perimeter Security

- 🌲 Wildlife & Forest Monitoring

- 🏢 Campus & Building Security
""")

    st.markdown("---")

    st.info(
        "This application is developed for educational and research purposes to demonstrate AI-powered surveillance using RGB and Thermal Vision."
    )

# ----------------------------------------------------
# RGB LIVE SURVEILLANCE
# ----------------------------------------------------

elif option == "📷 RGB Live Surveillance":

    st.header("📷 RGB Live Surveillance")

    st.info("""
This module performs **real-time object detection and tracking** using your system webcam.

Click the button below to launch the desktop surveillance application.
""")

    if st.button("▶ Launch Live Surveillance"):

        os.system(
    '"C:\\Users\\Swapnil\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" rgb_surveillance.py'
)

# ----------------------------------------------------
# RGB VIDEO ANALYSIS
# ----------------------------------------------------

elif option == "🎯 RGB Video Analysis":

    rgb_tracking()

# ----------------------------------------------------
# THERMAL IMAGE DETECTION
# ----------------------------------------------------

elif option == "🔥 Thermal Human Detection":

    thermal_detection()

# ----------------------------------------------------
# THERMAL VIDEO TRACKING
# ----------------------------------------------------

elif option == "🔥 Thermal Video Tracking":

    thermal_tracking()