# deepvision_streamlit.py
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

st.set_page_config(
    page_title="DeepVision - Crowd Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS (kept same-ish)
# ---------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.0rem;
        background: linear-gradient(45deg, #1f77b4, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        animation: pulse 2s infinite;
    }
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
    }
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown('<h1 class="main-header">🧠 DeepVision AI</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Crowd Monitoring & Analytics Platform")

# ---------------------------
# Sidebar - Controls
# ---------------------------
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    input_type = st.radio("📸 Select Input Type:", ["Live Camera", "Video Upload", "Image Upload"])
    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    # Note: detection_confidence (named scaleFactor) is float; give reasonable defaults
    crowd_threshold = st.slider("Crowd Alert Threshold", 1, 50, 7)
    detection_confidence = st.slider("Detection Sensitivity (scaleFactor)", 1.05, 1.5, 1.1, 0.01)
    min_neighbors = st.slider("Detection Quality (minNeighbors)", 3, 10, 5)
    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    enable_audio = st.checkbox("Enable Audio Alerts", value=True)
    enable_visual = st.checkbox("Enable Visual Alerts", value=True)
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.markdown(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    auto_save_csv = st.checkbox("Auto-save analytics CSV to disk", value=False)
    st.markdown("---")
    st.markdown("### 🎯 Quick Tips")
    st.info("""
    - Good lighting improves detection
    - Camera at eye/face height works better
    - Adjust sensitivity and neighbors based on environment
    """)

# ---------------------------
# Layout Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🎥 Live Monitoring", "📊 Analytics", "ℹ️ About"])

# ---------------------------
# Initialize session state
# ---------------------------
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'crowd_data' not in st.session_state:
    st.session_state.crowd_data = []
if 'stop_video' not in st.session_state:
    st.session_state.stop_video = False

# ---------------------------
# Load Haar Cascade
# ---------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# Helper: save analytics CSV
# ---------------------------
def save_analytics_csv(df):
    fname = f"crowd_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    return fname

# ---------------------------
# Tab 1: Live Monitoring
# ---------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    metric_current = col1.metric("Current Count", 0, delta="0")
    col2.metric("Alert Threshold", crowd_threshold)
    col3.metric("Detection Level", f"{detection_confidence:.2f}")
    status_placeholder = col4.empty()
    status_placeholder.markdown('<div class="success-card">**Status:** Ready<br>**System:** Idle</div>', unsafe_allow_html=True)

    st.markdown("### 🖼️ Live Feed")
    FRAME_WINDOW = st.empty()
    alert_display = st.empty()

    # Controls for modes
    if input_type == "Live Camera":
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("🎬 Start Camera"):
                st.session_state.monitoring = True
        with c2:
            if st.button("🛑 Stop Camera"):
                st.session_state.monitoring = False
        with c3:
            if st.button("🗑️ Clear History"):
                st.session_state.alert_history = []
                st.session_state.crowd_data = []
                st.success("History cleared!")

        # Start camera loop
        if st.session_state.monitoring:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access camera. Check permissions or camera usage by other apps.")
                st.session_state.monitoring = False
            else:
                status_placeholder.markdown('<div class="info-card">**Status:** 🔴 Monitoring Active<br>**Camera:** Running</div>', unsafe_allow_html=True)
                try:
                    while st.session_state.monitoring:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Failed to capture frame from camera.")
                            break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=float(detection_confidence),
                            minNeighbors=int(min_neighbors),
                            minSize=(30, 30)
                        )
                        crowd_count = len(faces)

                        # Update metrics
                        metric_current.metric("Current Count", crowd_count, delta=str(crowd_count) if crowd_count>0 else "0")

                        # Draw rectangles and record face-centers for heatmap
                        centers = []
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cx = x + w//2
                            cy = y + h//2
                            centers.append((cx, cy))

                        # Overlay basic density heatmap (approx)
                        if len(centers) > 0:
                            # compute a simple 2D histogram on the positions scaled to a small grid
                            pts = np.array(centers)
                            h_bins, w_bins = 40, 60
                            heatmap, _, _ = np.histogram2d(pts[:,1], pts[:,0], bins=[h_bins, w_bins], range=[[0, frame.shape[0]],[0, frame.shape[1]]])
                            # normalize to 0-255
                            heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
                            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                            display_frame = overlay
                        else:
                            display_frame = frame

                        # Info overlay
                        cv2.putText(display_frame, f"Crowd Count: {crowd_count}", (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(display_frame, f"Threshold: {crowd_threshold}", (20, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        status_color = (0, 255, 0) if crowd_count < crowd_threshold else (0, 0, 255)
                        status_text = "NORMAL" if crowd_count < crowd_threshold else "ALERT"
                        cv2.putText(display_frame, f"Status: {status_text}", (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                        # Handle alerts
                        if crowd_count >= crowd_threshold:
                            if enable_visual:
                                alert_display.markdown('<div class="alert-card">🚨 <b>OVERLOAD ALERT!</b> {} detected (Threshold: {})</div>'.format(crowd_count, crowd_threshold), unsafe_allow_html=True)
                            # log alert
                            st.session_state.alert_history.append({
                                'timestamp': datetime.now(),
                                'count': crowd_count,
                                'threshold': crowd_threshold,
                                'type': 'Overcrowding'
                            })
                            if enable_audio:
                                st.markdown("""
                                    <audio autoplay>
                                    <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                                    </audio>
                                    """, unsafe_allow_html=True)
                        else:
                            alert_display.markdown('<div class="success-card">✅ <b>CROWD NORMAL</b> {} detected</div>'.format(crowd_count), unsafe_allow_html=True)

                        # Store crowd data
                        st.session_state.crowd_data.append({
                            'timestamp': datetime.now(),
                            'count': crowd_count,
                            'status': status_text
                        })

                        # Show frame
                        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                        # tiny sleep for CPU stabilization
                        time.sleep(0.03)
                except Exception as e:
                    st.error(f"Error during camera loop: {e}")
                finally:
                    cap.release()
                    status_placeholder.markdown('<div class="success-card">**Status:** Ready<br>**System:** Idle</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Click **'Start Camera'** to begin real-time monitoring")
            FRAME_WINDOW.image("https://via.placeholder.com/800x450/2c3e50/ffffff?text=DeepVision+AI+Camera+Feed+Ready", use_column_width=True)

    # -----------------------------
    # Video Upload Mode
    # -----------------------------
    elif input_type == "Video Upload":
        uploaded_file = st.file_uploader("🎞️ Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("▶️ Process Video"):
                    # start processing
                    st.session_state.stop_video = False
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    cap = cv2.VideoCapture(tfile.name)
                    status_placeholder.markdown('<div class="info-card">**Status:** 🎬 Processing Video<br>**File:** {}</div>'.format(uploaded_file.name), unsafe_allow_html=True)
                    try:
                        while cap.isOpened() and not st.session_state.stop_video:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=float(detection_confidence), minNeighbors=int(min_neighbors), minSize=(30,30))
                            crowd_count = len(faces)
                            metric_current.metric("Current Count", crowd_count)
                            for (x,y,w,h) in faces:
                                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                            cv2.putText(frame, f"Crowd Count: {crowd_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                            if crowd_count >= crowd_threshold:
                                alert_display.markdown('<div class="alert-card">🚨 <b>OVERLOAD ALERT!</b> {} detected</div>'.format(crowd_count), unsafe_allow_html=True)
                                st.session_state.alert_history.append({
                                    'timestamp': datetime.now(),
                                    'count': crowd_count,
                                    'threshold': crowd_threshold,
                                    'type': 'Overcrowding'
                                })
                            else:
                                alert_display.markdown('<div class="success-card">✅ Crowd Normal</div>', unsafe_allow_html=True)
                            st.session_state.crowd_data.append({'timestamp': datetime.now(), 'count': crowd_count, 'status': "ALERT" if crowd_count>=crowd_threshold else "NORMAL"})
                            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                            time.sleep(0.03)
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    finally:
                        cap.release()
                        try:
                            os.unlink(tfile.name)
                        except:
                            pass
                        status_placeholder.markdown('<div class="success-card">**Status:** Ready<br>**System:** Idle</div>', unsafe_allow_html=True)
            with c2:
                if st.button("⏹️ Stop Processing"):
                    st.session_state.stop_video = True

    # -----------------------------
    # Image Upload Mode
    # -----------------------------
    elif input_type == "Image Upload":
        uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg","jpeg","png"])
        if uploaded_image is not None:
            if st.button("🔍 Analyze Image"):
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=float(detection_confidence), minNeighbors=int(min_neighbors), minSize=(30,30))
                crowd_count = len(faces)
                for (x,y,w,h) in faces:
                    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)
                    cv2.putText(image, 'Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(image, f"Crowd Count: {crowd_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.metric("People Detected", crowd_count)
                st.metric("Alert Threshold", crowd_threshold)
                if crowd_count >= crowd_threshold:
                    st.error("🚨 Overcrowding Alert!")
                    st.session_state.alert_history.append({'timestamp': datetime.now(), 'count': crowd_count, 'threshold': crowd_threshold, 'type': 'Overcrowding'})
                else:
                    st.success("✅ Crowd Level Normal")

# ---------------------------
# Tab 2: Analytics & Reports
# ---------------------------
with tab2:
    st.markdown("## 📈 Analytics & Reports")
    if st.session_state.crowd_data:
        df = pd.DataFrame(st.session_state.crowd_data)
        # normalize timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        colA, colB = st.columns([2,1])

        with colA:
            st.markdown("### 📋 Alert / Crowd History")
            st.dataframe(df.sort_values('timestamp', ascending=False).reset_index(drop=True), use_container_width=True)

        with colB:
            st.markdown("### 📊 Summary")
            total_records = len(df)
            avg_count = df['count'].mean() if 'count' in df.columns else 0
            max_count = int(df['count'].max()) if 'count' in df.columns else 0
            st.metric("Total Samples", total_records)
            st.metric("Average Count", f"{avg_count:.1f}")
            st.metric("Maximum Count", max_count)

            if st.session_state.alert_history:
                st.markdown("### ⚠️ Alerts Recorded")
                alerts_df = pd.DataFrame(st.session_state.alert_history)
                st.write(f"Total Alerts: {len(alerts_df)}")
                if st.button("Export Alerts CSV"):
                    csv = alerts_df.to_csv(index=False)
                    st.download_button("Download Alerts CSV", data=csv, file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

            # Auto-save if requested
            if auto_save_csv:
                fname = save_analytics_csv(df)
                st.info(f"Analytics auto-saved to {fname}")

        # Live plot (matplotlib)
        st.markdown("### ⏱️ Crowd Count Over Time (Live)")
        fig, ax = plt.subplots(figsize=(8,3))
        try:
            ax.plot(df['timestamp'], df['count'])
            ax.set_ylabel("Count")
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not plot live chart: {e}")

        # Download full analytics
        csv = df.to_csv(index=False)
        st.download_button("Download Full Analytics CSV", data=csv, file_name=f"crowd_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    else:
        st.info("📊 No analytics data available yet. Start monitoring to collect data.")

# ---------------------------
# Tab 3: About
# ---------------------------
with tab3:
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("## 🏢 About DeepVision AI")
        st.markdown("""
        DeepVision AI is an intelligent crowd monitoring system that uses computer vision 
        to detect and analyze crowd density in real-time.
        
        ### 🎯 Key Features:
        - **Real-time Monitoring**: Live camera feed analysis
        - **Multiple Input Sources**: Camera, video files, and images
        - **Smart Alerts**: Configurable threshold-based notifications
        - **Analytics Dashboard**: Historical data and insights
        - **User-friendly Interface**: Intuitive controls and visualizations
        """)
        st.markdown("### 🔧 Technology Stack:\n- Streamlit\n- OpenCV\n- Haar Cascade (face detection)\n- Pandas / NumPy")
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.info("1. Select input type\n2. Adjust detection settings\n3. Start monitoring\n4. View analytics")
        st.markdown("### 📞 Support")
        st.success("Email: support@deepvision.ai\nWebsite: www.deepvision.ai")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>🧠 <b>DeepVision AI Crowd Monitoring System</b> | Built with ❤️ using Streamlit & OpenCV | © 2024 DeepVision AI.</div>", unsafe_allow_html=True)
