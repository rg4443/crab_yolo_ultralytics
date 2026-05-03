import cv2
from ultralytics import YOLO
import threading
import numpy as np
import time
import csv # For logging

class Frame:
    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.last_update = time.time() 
        self.latency = 0.0             # For Performance Metrics

    def set(self, frame, start_time):
        with self.lock:
            self.frame = frame.copy()
            self.last_update = time.time()
            self.latency = time.time() - start_time

    def get(self):
        with self.lock:
            return self.frame.copy(), self.latency
        
def telemetry_logger(frame_list, filename="vision_performance.csv"):
    """
    Background worker that samples system performance every 10 seconds.
    """
    print(f"[Telemetry] Logging started. Saving to {filename}")
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Cam0_Latency", "Cam1_Latency", "Cam2_Latency", "Cam3_Latency"])

    while not interrupt:
        time.sleep(10) 
        
        timestamp = time.strftime("%H:%M:%S")
        current_latencies = [f.get()[1] for f in frame_list]
        
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp] + current_latencies)
            
    print("[Telemetry] Logging stopped.")

def combine(imgs):
    img1 = cv2.resize(imgs[0], (1920, 1080))
    img2 = cv2.resize(imgs[1], (640, 360))
    img3 = cv2.resize(imgs[2], (640, 360))
    img4 = cv2.resize(imgs[3], (640, 360))
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

interrupt = False
model = YOLO("runs/detect/train6/weights/best.pt")

def run_camera(url, frame_obj, model=None, camera_id=0):
    global interrupt

    retry_count = 0

    while not interrupt:
        print(f"[Executive] Initializing Stream {camera_id}...")

        wait_time = min(retry_count * 2, 10) 
        if retry_count > 0:
            print(f"[Executive] Retry {retry_count} for Stream {camera_id} in {wait_time}s...")
            time.sleep(wait_time)

        video = cv2.VideoCapture(url)
        
        # Heartbeat tracking
        last_heartbeat = time.time()
        timeout_threshold = 2.0  # If no frames for 2 seconds, it's a "Stall"

        while not interrupt:
            r, f = video.read()
            start_time = time.time()
            
            # Only update the heartbeat if the read was successful
            if r and f is not None:
                last_heartbeat = time.time()
                
                if model and camera_id == 0:
                    results = model.predict(f, verbose=False)
                    f = results[0].plot()

                    # Count Crabs on Upper Left Corner
                    number = len(results[0].boxes)
                    cv2.putText(f, f"Green Crabs Detected: {number}", (7, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # Metadata Overlay
                    cv2.putText(f, f"LATENCY: {frame_obj.latency:.3f}s", (7, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                frame_obj.set(f, start_time)

            if (time.time() - last_heartbeat) > timeout_threshold:
                print(f"[Watchdog] Stream {camera_id} HEARTBEAT LOST. Auto-Recovering...")
                break 
                
        video.release()
        time.sleep(1) 

frames = []
threads = []
urls = [
    "udp://192.168.2.1:1984?fifo_size=1000000&overrun_nonfatal=1"
    "udp://192.168.2.1:1985?overrun_nonfatal=1", 
    "udp://192.168.2.1:1986?overrun_nonfatal=1", 
    "udp://192.168.2.1:1987?overrun_nonfatal=1",
]

# Initialize Threads
for i in range(4):
    frames.append(Frame())
    target_model = model if i == 0 else None
    threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i], target_model, i)))
    threads[i].daemon = True # Ensures threads exit when main loop does
    threads[i].start()

log_thread = threading.Thread(target=telemetry_logger, args=(frames,))
log_thread.daemon = True
log_thread.start()

print("[System] All Vision Threads Active.")

while not interrupt:
    try:
        raw_data = [f.get() for f in frames]
        imgs = [data[0] for data in raw_data]
        latencies = [data[1] for data in raw_data]

        if len(imgs) == 4: 
            combined = combine(imgs)
            cv2.imshow('Slugbotics Flight Deck', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): interrupt = True

    except Exception as e:
        print(f'[CRITICAL ERROR] {e}')
        interrupt = True

cv2.destroyAllWindows()