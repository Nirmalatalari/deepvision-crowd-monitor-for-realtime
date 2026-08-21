DeepVision — Crowd Monitor AI

AI-based crowd monitoring system for real-time density estimation and overcrowding detection using computer vision.

Built with Streamlit, OpenCV, and YOLOv8. The app analyzes video feeds to count people/faces and sends email alerts when a crowd threshold is exceeded.

Features
Crowd/face detection — YOLOv8 (ultralytics) on GPU when available, with automatic fallback to OpenCV's Haar Cascade on CPU
Live video analysis — upload a video and process it frame-by-frame for crowd counting over time
Overcrowding alerts — sends an email notification when the crowd count crosses a configurable threshold
Setup
1. Clone the repository
bash
git clone https://github.com/<your-username>/deepvision-crowd-monitor.git
cd deepvision-crowd-monitor
2. Install dependencies
bash
pip install -r requirements.txt
3. Configure email alerts (optional)

Set these as environment variables — never hardcode credentials in the code:

bash
export EMAIL_SENDER="your-email@gmail.com"
export EMAIL_RECEIVER="alert-recipient@gmail.com"
export EMAIL_PASSWORD="your-app-password"

Use a Gmail App Password, not your regular account password.

4. Run the app
bash
streamlit run app.py

Then open http://localhost:8501 in your browser.

Tech stack

Python, Streamlit, YOLOv8 (Ultralytics), OpenCV, PyTorch
