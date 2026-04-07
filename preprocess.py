import cv2
import numpy as np
import os
import time
import platform
import sys
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    'crowd_threshold': 7,
    'alert_cooldown': 5,  # seconds
    'min_face_size': (30, 30),
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'fps': 20,
    'heatmap_alpha': 0.3,
    'detection_confidence': 0.7,
    'frame_skip': 2,  # Process every nth frame for performance
    'use_dnn_detector': False  # Set to True if you have DNN model files
}

# ----------------------------
# Cross-platform audio alert
# ----------------------------
def play_alert():
    """Play alert sound compatible with Windows, macOS, and Linux"""
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.Beep(1000, 1000)  # Frequency=1000Hz, Duration=1s
        elif system == "Darwin":  # macOS
            os.system('afplay /System/Library/Sounds/Ping.aiff 2>/dev/null')
        else:  # Linux
            os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null')
    except Exception as e:
        print(f"Audio alert failed: {e}")
        print("⚠ ALERT: Crowd limit exceeded! (visual only)")

# ----------------------------
# Advanced Density Map
# ----------------------------
def create_advanced_density_map(faces, frame_shape):
    """Create a more accurate density map based on face sizes"""
    h, w = frame_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for (x, y, w_face, h_face) in faces:
        cx, cy = x + w_face // 2, y + h_face // 2
        # Create Gaussian kernel based on face size (larger faces have more influence)
        radius = max(w_face, h_face) // 2
        cv2.circle(density_map, (cx, cy), radius, 1, -1)
    
    # Apply Gaussian blur with adaptive kernel size
    kernel_size = max(3, int(min(h, w) * 0.02) | 1)  # Ensure odd number, minimum 3
    density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), 0)
    
    # Normalize
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    return density_map

# ----------------------------
# DNN Face Detector (Optional)
# ----------------------------
def setup_dnn_detector():
    """Initialize DNN-based face detector if model files are available"""
    try:
        # You need to download these model files:
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector_uint8.pb
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
        model_path = "models/"
        if os.path.exists(os.path.join(model_path, "opencv_face_detector_uint8.pb")):
            net = cv2.dnn.readNetFromTensorflow(
                os.path.join(model_path, "opencv_face_detector_uint8.pb"),
                os.path.join(model_path, "opencv_face_detector.pbtxt")
            )
            print("DNN face detector loaded successfully")
            return net
        else:
            print("DNN model files not found. Using Haar cascade detector.")
            return None
    except Exception as e:
        print(f"DNN detector setup failed: {e}. Using Haar cascade.")
        return None

def detect_faces_dnn(frame, net):
    """Detect faces using DNN model"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIG['detection_confidence']:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                faces.append((x1, y1, width, height))
    
    return faces

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Use video file instead of webcam
    video_file = "video.mp4"
    
    # Check if video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        print("Please make sure 'video.mp4' exists in the current directory.")
        return
    
    # Initialize video capture from file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_file}'.")
        print("The file might be corrupted or in an unsupported format.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video loaded: {video_file}")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {original_fps:.2f}")
    
    # Setup output
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/face_density_{os.path.splitext(video_file)[0]}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, CONFIG['fps'], (width, height))
    
    # Initialize face detector
    if CONFIG['use_dnn_detector']:
        dnn_net = setup_dnn_detector()
        use_dnn = dnn_net is not None
    else:
        use_dnn = False
        dnn_net = None
    
    if not use_dnn:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load Haar cascade classifier.")
            cap.release()
            return
    
    print("Starting face counting on video file")
    print(f"Crowd threshold: {CONFIG['crowd_threshold']}")
    print(f"Using {'DNN' if use_dnn else 'Haar Cascade'} detector")
    print("Press 'q' to quit, 'r' to reset alert cooldown")
    print("Press 'p' to pause/resume, '+'/'-' to adjust threshold")
    
    # Performance tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    # Alert system
    last_alert_time = 0
    alert_triggered = False
    
    # Frame skipping for performance
    frame_counter = 0
    processed_faces = []
    
    # Video control
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    break
                
                frame_counter += 1
                
                # Process face detection every nth frame for performance
                if frame_counter % CONFIG['frame_skip'] == 0:
                    if use_dnn and dnn_net is not None:
                        # Use DNN detector
                        processed_faces = detect_faces_dnn(frame, dnn_net)
                    else:
                        # Use Haar cascade detector
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        processed_faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=CONFIG['scale_factor'],
                            minNeighbors=CONFIG['min_neighbors'],
                            minSize=CONFIG['min_face_size']
                        )
                
                faces = processed_faces
                count = len(faces)
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw face rectangles and centers
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cx, cy = x + w // 2, y + h // 2
                    cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)
                    # Display confidence for DNN (if available)
                    if use_dnn:
                        cv2.putText(display_frame, f"{w}x{h}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Create and overlay density map
                if count > 0:
                    density_map = create_advanced_density_map(faces, frame.shape)
                    heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(display_frame, 1-CONFIG['heatmap_alpha'], 
                                            heatmap, CONFIG['heatmap_alpha'], 0)
                else:
                    overlay = display_frame
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_frame_count
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Display information
                cv2.putText(overlay, f"Human Count: {count}", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(overlay, f"FPS: {current_fps}", (width - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Frame: {frame_counter}/{total_frames}", (width - 250, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"Threshold: {CONFIG['crowd_threshold']}", (30, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Alert system
                if count >= CONFIG['crowd_threshold']:
                    cv2.putText(overlay, "⚠ ALERT: Crowd limit exceeded!", (30, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Draw red border for alert
                    cv2.rectangle(overlay, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                    
                    # Trigger sound alert with cooldown
                    current_time = time.time()
                    if current_time - last_alert_time > CONFIG['alert_cooldown']:
                        play_alert()
                        last_alert_time = current_time
                        alert_triggered = True
                        print(f"ALERT triggered at frame {frame_counter} - Count: {count}")
                else:
                    alert_triggered = False
                
                # Display and save
                cv2.imshow("Advanced Face Counting + Density", overlay)
                out.write(overlay)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('r'):
                # Reset alert cooldown
                last_alert_time = 0
                print("Alert cooldown reset")
            elif key == ord('+'):
                CONFIG['crowd_threshold'] += 1
                print(f"Threshold increased to: {CONFIG['crowd_threshold']}")
            elif key == ord('-') and CONFIG['crowd_threshold'] > 1:
                CONFIG['crowd_threshold'] -= 1
                print(f"Threshold decreased to: {CONFIG['crowd_threshold']}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved at: {output_path}")
        print(f"Processed {frame_counter} frames")
        print("Application closed successfully")

if __name__ == "__main__":
    main()