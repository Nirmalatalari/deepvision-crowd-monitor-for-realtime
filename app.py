import sys
print(sys.executable)
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import tempfile
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import torch
import GPUtil
from ultralytics import YOLO
import psutil
import json
from typing import Optional

# -----------------------------------
# 🚀 GPU OPTIMIZATION & PERFORMANCE MONITORING
# -----------------------------------

class GPUMonitor:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu_available else 'cpu')

    def get_gpu_info(self) -> Optional[dict]:
        """Get GPU utilization and memory information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'name': gpu.name,
                    'load': f"{gpu.load*100:.1f}%",
                    'memory_used': f"{gpu.memoryUsed}MB",
                    'memory_total': f"{gpu.memoryTotal}MB",
                    'temperature': f"{gpu.temperature}°C"
                }
        except Exception:
            # silently ignore GPUtil errors
            pass
        return None

    def get_system_info(self) -> dict:
        """Get system resource utilization"""
        return {
            'cpu_usage': f"{psutil.cpu_percent()}%",
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'gpu_info': self.get_gpu_info()
        }

# Initialize GPU monitor
gpu_monitor = GPUMonitor()

# -----------------------------------
# 🎯 ENHANCED FACE DETECTION WITH YOLO
# -----------------------------------

class EnhancedFaceDetector:
    def __init__(self, model_type: str = 'yolo'):
        self.model_type = model_type
        self.device = gpu_monitor.device
        self.model = None
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.load_model()

    def load_model(self):
        """Load the appropriate face detection model"""
        try:
            if self.model_type == 'yolo' and gpu_monitor.gpu_available:
                # Use local model if present, else ultralytics will download
                # NOTE: Reverting to yolov8n.pt as per the updated code provided.
                # Remember that using a face-specific model (like yolov8n-face.pt) is better for accuracy!
                model_path = "yolov8n.pt"
                self.model = YOLO(model_path)
                st.success(f"🚀 YOLO model loaded on {self.device}")
            else:
                # If model_type 'yolo' but GPU not available, fall back to Haar
                if self.model_type == 'yolo':
                    st.info("⚠️ YOLO requested but no GPU detected — falling back to Haar Cascade (CPU).")
                else:
                    st.info("📱 Using Haar Cascade (CPU mode)")
                self.model = None
        except Exception as e:
            st.warning(f"⚠️ YOLO model not available, using Haar Cascade. Error: {e}")
            self.model = None

    def detect_faces(self, image: np.ndarray):
        """Detect faces using the selected model"""
        if self.model and gpu_monitor.gpu_available and self.model_type == 'yolo':
            return self.detect_faces_yolo(image)
        else:
            return self.detect_faces_haar(image)

    def detect_faces_yolo(self, image: np.ndarray):
        """Detect faces using YOLO (more accurate, GPU-accelerated)"""
        try:
            # YOLO expects RGB images (HWC)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # CONF=0.45 is a decent starting point, but consider lowering/raising this
            results = self.model(img_rgb, conf=0.45, verbose=False)  # returns list of results

            face_count = 0
            out_image = image.copy()

            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue

                # boxes.xyxy, boxes.conf, boxes.cls are tensors
                try:
                    xyxy = boxes.xyxy.cpu().numpy()  # Nx4
                    confs = boxes.conf.cpu().numpy()  # N
                    classes = boxes.cls.cpu().numpy()  # N
                except Exception:
                    # fallback: try converting via numpy directly
                    xyxy = np.array(boxes.xyxy)
                    confs = np.array(boxes.conf)
                    classes = np.array(boxes.cls)

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                    # class 0 in COCO = person
                    if int(cls) == 0:
                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(out_image, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        label = f"Person: {conf:.2f}"
                        cv2.putText(out_image, label, (x1i, max(y1i - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        face_count += 1

            return face_count, out_image

        except Exception as e:
            st.error(f"YOLO detection error: {e}")
            # fallback
            return self.detect_faces_haar(image)

    def detect_faces_haar(self, image: np.ndarray):
        """Detect faces using Haar Cascade (CPU fallback)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        out = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return len(faces), out

# -----------------------------------
# 🔔 ENHANCED EMAIL ALERT FUNCTION
# -----------------------------------

def get_smtp_password():
    # Prefer Streamlit secrets, then environment variable
    try:
        # expects: st.secrets["email"]["password"]
        return st.secrets["email"]["password"]
    except Exception:
        return os.environ.get("EMAIL_PASSWORD")

def send_email_alert(crowd_count, location="Main Entrance", threshold=1):
    try:
        # Email configuration - do NOT hardcode credentials in code; use secrets or env
        sender_email = os.environ.get("EMAIL_SENDER", "bvmadhav576@gmail.com")
        receiver_email = os.environ.get("EMAIL_RECEIVER", sender_email)
        sender_password = get_smtp_password()

        if not sender_password:
            st.warning("⚠️ Email password not configured. Set st.secrets['email']['password'] or EMAIL_PASSWORD env var.")
            return False

        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "⚠️ DEEPVISION ALERT TRIGGERED!"

        # Enhanced email body with system info
        system_info = gpu_monitor.get_system_info()
        gpu_info = system_info.get('gpu_info') if system_info else None

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #ff0000;">⚠️ DEEPVISION ALERT TRIGGERED!</h2>
            <p><strong>Crowd Monitoring System</strong> has detected an overcrowding situation.</p>
            <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #ff0000;">
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Detected Count:</strong> {crowd_count}</p>
                <p><strong>Threshold:</strong> {threshold}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>System Mode:</strong> {'GPU Accelerated' if gpu_monitor.gpu_available else 'CPU Mode'}</p>
            </div>
            <div style="background-color: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4>System Status:</h4>
                <p>CPU Usage: {system_info['cpu_usage']} | Memory: {system_info['memory_usage']}</p>
                {f"<p>GPU: {gpu_info['name']} | Load: {gpu_info['load']} | Memory: {gpu_info['memory_used']}/{gpu_info['memory_total']}</p>" if gpu_info else "<p>GPU: Not Available</p>"}
            </div>
            <p style="color: #ff0000; font-weight: bold;">Please take immediate action to ensure safety.</p>
            <br>
            <p>Best regards,<br>DeepVision Monitoring System</p>
        </body>
        </html>
        """

        message.attach(MIMEText(body, "html"))

        # Send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)
        server.quit()

        st.success("📧 Email alert sent successfully!")
        return True

    except Exception as e:
        st.error(f"❌ Email Error: {e}")
        return False

# -----------------------------------
# 🎨 CUSTOM CSS FOR BACKGROUND COLOR
# -----------------------------------
def set_background_color(color_code):
    """Injects CSS to set the application background color."""
    css = f"""
    <style>
    .stApp {{
        background-color: {color_code};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------------
# 🐳 DOCKER-OPTIMIZED STREAMLIT APP
# -----------------------------------

st.set_page_config(page_title="DeepVision - Crowd Monitoring", layout="wide")

# Inject the background color
# You can change '#f0f2f6' to any hex color code (e.g., '#262730' for dark grey)
set_background_color("#f0f2f6") 

st.title(" DeepVision - Crowd Monitoring System")
st.markdown("###  Dockerized &  GPU-Optimized Real-Time Monitoring")

# System info sidebar
with st.sidebar:
    st.header("🔧 System Information")

    # Display GPU/CPU info
    system_info = gpu_monitor.get_system_info()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", system_info['cpu_usage'])
        st.metric("Memory Usage", system_info['memory_usage'])

    with col2:
        if system_info['gpu_info']:
            st.metric("GPU Load", system_info['gpu_info']['load'])
            st.metric("GPU Memory", system_info['gpu_info']['memory_used'])
        else:
            st.info("🔍 No GPU detected")

    st.markdown("---")

    # Model selection
    model_choice = st.radio(
        "🤖 Detection Model:",
        ["YOLO (GPU Recommended)", "Haar Cascade (CPU)"],
        index=0 if gpu_monitor.gpu_available else 1
    )

    model_type = 'yolo' if model_choice == "YOLO (GPU Recommended)" else 'haar'

    # Application controls
    input_type = st.radio("📸 Select Input Type:", ["Live Camera", "Video Upload", "Image Upload"])
    crowd_threshold = st.slider("⚙️ Crowd Limit", 1, 50, 4)
    alert_location = st.text_input("📍 Alert Location", "Main Entrance")

    # Alert configuration
    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    enable_email = st.checkbox("Enable Email Alerts", value=True)

    # Performance settings
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    processing_delay = st.slider("Processing Delay (ms)", 10, 200, 50, help="Lower for faster processing")

# Initialize detector (cached resource)
@st.cache_resource
def load_detector(mt):
    return EnhancedFaceDetector(model_type=mt)

detector = load_detector(model_type)

FRAME_WINDOW = st.image([])
status_text = st.empty()
performance_text = st.empty()
alert_log = st.empty()

# session_state initializations
if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = 0.0
if 'camera_stop' not in st.session_state:
    st.session_state['camera_stop'] = False

# ======================================================
# 🎥 ENHANCED LIVE CAMERA MODE WITH GPU
# ======================================================
if input_type == "Live Camera":
    start_btn = st.button("🎥 Start Camera")
    stop_btn = st.button("🛑 Stop Camera")

    if start_btn:
        st.session_state['camera_stop'] = False
    if stop_btn:
        st.session_state['camera_stop'] = True

    if not st.session_state['camera_stop'] and start_btn:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        st.info("📷 Camera started with GPU acceleration!" if gpu_monitor.gpu_available else "📷 Camera started (CPU mode)")

        frame_count = 0
        start_time = time.time()

        try:
            while cap.isOpened() and not st.session_state['camera_stop']:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform face detection
                detection_start = time.time()
                crowd_count, processed_frame = detector.detect_faces(frame)
                detection_time = (time.time() - detection_start) * 1000

                # Add performance metrics to frame
                cv2.putText(processed_frame, f"Crowd Count: {crowd_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(processed_frame, f"Threshold: {crowd_threshold}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Model: {model_type.upper()}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Device: {'GPU' if gpu_monitor.gpu_available else 'CPU'}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Alert system with cooldown
                current_time = time.time()
                if crowd_count >= crowd_threshold:
                    status_text.error(f"🚨 Overcrowded! {crowd_count} people detected at {alert_location}")

                    # Check cooldown period
                    if current_time - st.session_state['last_alert_time'] > 30:
                        # Trigger email alert
                        if enable_email:
                            send_email_alert(crowd_count, alert_location, crowd_threshold)

                        st.session_state['last_alert_time'] = current_time

                        # Log the alert
                        alert_time = datetime.now().strftime("%H:%M:%S")
                        alert_log.warning(f"📝 Alert triggered at {alert_time}: {crowd_count} people detected")

                        # Audio alert
                        st.markdown(
                            """
                            <audio autoplay>
                            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                            </audio>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    status_text.success(f"✅ Crowd OK ({crowd_count} people)")

                # Update performance metrics
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                performance_text.info(
                    f"⚡ Performance: {fps:.1f} FPS | "
                    f"Detection: {detection_time:.1f}ms | "
                    f"Model: {model_type} | "
                    f"Device: {'GPU' if gpu_monitor.gpu_available else 'CPU'}"
                )

                #FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                processed_frame = cv2.resize(processed_frame, (1280, 720))
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                # allow other Streamlit events to process
                time.sleep(processing_delay / 1000.0)
        except Exception as e:
            st.error(f"Camera error: {e}")
        finally:
            cap.release()
            FRAME_WINDOW.empty()
            status_text.info("🛑 Camera stopped.")
    else:
        st.info("👆 Press 'Start Camera' to begin monitoring.")

# ======================================================
# 🎞️ VIDEO UPLOAD MODE
# ======================================================
elif input_type == "Video Upload":
    uploaded_file = st.file_uploader("🎞️ Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        play = st.button("▶️ Play Video")
        stop_video = st.button("🛑 Stop Video")
        if 'video_stop' not in st.session_state:
            st.session_state['video_stop'] = False

        if play:
            st.session_state['video_stop'] = False
        if stop_video:
            st.session_state['video_stop'] = True

        if play:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            fps_text = st.empty()

            frame_count = 0
            start_time = time.time()

            try:
                while cap.isOpened() and not st.session_state['video_stop']:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    crowd_count, processed_frame = detector.detect_faces(frame)

                    cv2.putText(processed_frame, f"Crowd Count: {crowd_count}", (20, 40),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Alert system
                    current_time = time.time()
                    if crowd_count >= crowd_threshold:
                        status_text.error(f"🚨 Overcrowded! {crowd_count} people detected")

                        if current_time - st.session_state['last_alert_time'] > 30:
                            if enable_email:
                                send_email_alert(crowd_count, alert_location, crowd_threshold)

                            st.session_state['last_alert_time'] = current_time

                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                    fps_text.text(f"📊 Video FPS: {fps:.1f} | Detected: {crowd_count} people")

                    stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                    time.sleep(processing_delay / 1000.0)
            except Exception as e:
                st.error(f"Video playback error: {e}")
            finally:
                cap.release()
                try:
                    os.remove(tfile.name)
                except Exception:
                    pass
                st.info("🛑 Video stopped.")

# ======================================================
# 🖼️ IMAGE UPLOAD MODE
# ======================================================
elif input_type == "Image Upload":
    uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        detection_start = time.time()
        crowd_count, processed_image = detector.detect_faces(image)
        detection_time = (time.time() - detection_start) * 1000

        cv2.putText(processed_image, f"Crowd Count: {crowd_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(processed_image, f"Detection Time: {detection_time:.1f}ms", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if crowd_count >= crowd_threshold:
            st.error(f"Overcrowded! {crowd_count} people detected at {alert_location}")

            # Send one-time email alert for image
            if enable_email:
                send_email_alert(crowd_count, alert_location, crowd_threshold)

        else:
            st.success(f"✅ Crowd OK ({crowd_count} people)")

        #st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")
        #processed_image = cv2.resize(processed_image, (640, 480))
        processed_image = cv2.resize(processed_image, (1280, 720))
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), channels="RGB")
        st.info(f"⏱️ Processing time: {detection_time:.1f}ms using {'GPU' if gpu_monitor.gpu_available else 'CPU'}")

# -----------------------------------
# 🐳 DOCKER DEPLOYMENT INSTRUCTIONS
# -----------------------------------
with st.sidebar.expander(" Docker Deployment"):
    st.markdown("""
    ### Quick Start:
    ```bash
    # Build and run with Docker Compose
    docker-compose up -d

    # Access the application
    http://localhost:8501

    # View logs
    docker-compose logs -f

    # Stop the application
    docker-compose down
    ```

    ### GPU Support (NVIDIA):
    ```bash
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L [https://nvidia.github.io/nvidia-docker/gpgkey](https://nvidia.github.io/nvidia-docker/gpgkey) | sudo apt-key add -
    curl -s -L [https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list](https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list) | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

    # Test GPU access
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
    ```
    """)

# Add startup message
if 'startup_shown' not in st.session_state:
    st.balloons()
    st.success(f" DeepVision Started Successfully! | Mode: {'GPU Accelerated' if gpu_monitor.gpu_available else 'CPU'} | Model: {model_type.upper()}")
    st.session_state.startup_shown = True