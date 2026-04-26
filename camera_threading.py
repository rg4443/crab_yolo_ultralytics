import cv2
from ultralytics import YOLO
import threading
import numpy as np

class Frame:
    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.lock = threading.Lock()
    def set(self, frame):
        with self.lock:
            self.frame = frame.copy()
    def get(self):
        with self.lock:
            return self.frame.copy()

def combine(imgs):
    img1 = cv2.resize(imgs[0], (1920, 1080))
    img2 = cv2.resize(imgs[1], (640, 360))
    img3 = cv2.resize(imgs[2], (640, 360))
    img4 = cv2.resize(imgs[3], (640, 360))
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

interrupt = False
model = YOLO("runs/detect/train6/weights/best.pt")

def run_camera(url, frame, model=None):
    try:
        video = cv2.VideoCapture(url)
        while not interrupt:
            r, f = video.read()
            if not r or f is None:
                continue
            if model is None:
                frame.set(f)
            else:
                results = model.predict(f)
                plotted = results[0].plot()

                # Count Crabs on Upper Left Corner
                number = len(results[0].boxes)
                cv2.putText(plotted, f"Green Crabs Detected: {number}", (7, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                frame.set(plotted)
        video.release()
    except Exception as e:
        print(f"ERROR: {e}")

frames = []
threads = []
urls = [
    0, 
    0, 
    0, 
    0,
]

for i in range(4):
    frames.append(Frame())
    if i == 0:
        threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i], model)))
    else:
        threads.append(threading.Thread(target=run_camera, args=(urls[i], frames[i])))
    threads[i].start()

while not interrupt:
    try:
        copies = [f.get() for f in frames]
        if len(copies) == 4: 
            combined = combine(copies)
            cv2.imshow('Frontend', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            interrupt = True
    except KeyboardInterrupt:
        interrupt = True
    except Exception as e:
        print(f'ERROR: {e}')
        interrupt = True

for t in threads:
    t.join()
cv2.destroyAllWindows()
print()