from ultralytics import YOLO
import cv2

def combine(img1, img2, img3, img4):
    img1 = cv2.resize(img1, (1920, 1080))
    img2 = cv2.resize(img2, (640, 360))
    img3 = cv2.resize(img3, (640, 360))
    img4 = cv2.resize(img4, (640, 360))
    img5 = cv2.hconcat([img2, img3, img4])
    return cv2.vconcat([img1, img5])

# video1 = cv2.VideoCapture("udp://192.168.2.1:1984?fifo_size=1000000&overrun_nonfatal=1") 
video2 = cv2.VideoCapture("udp://192.168.2.1:1985?overrun_nonfatal=1") 
video3 = cv2.VideoCapture("udp://192.168.2.1:1986?overrun_nonfatal=1") 
video4 = cv2.VideoCapture("udp://192.168.2.1:1987?overrun_nonfatal=1") 

# print("camera1 opened:", video1.isOpened())
print("camera2 opened:", video2.isOpened())
print("camera3 opened:", video3.isOpened())
print("camera4 opened:", video4.isOpened())

model = YOLO("runs/detect/train5/weights/best.pt")

while True:
    # ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    ret3, frame3 = video3.read()
    ret4, frame4 = video4.read()

    # if not ret1 or frame1 is None:
        # print("failed to read frame1")
        # continue

    if not ret2 or frame2 is None:
        print("failed to read frame2")
        continue
    if not ret3 or frame3 is None:
        print("failed to read frame3")
        continue
    if not ret4 or frame4 is None:
        print("failed to read frame4")
        continue

    combine_frame = combine(frame2, frame2, frame3, frame4)

    cv2.imshow("Crab detections", combine_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# video1.release()
video2.release()
video3.release()
video4.release()
cv2.destroyAllWindows()