from ultralytics import YOLO
import cv2

video = cv2.VideoCapture(0)
print("camera opened:", video.isOpened())

model = YOLO("runs/detect/train/weights/best.pt")

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        print("failed to read frame")
        break

    results = model.predict(frame, verbose=False)
    number = len(results[0].boxes)
    plotted = results[0].plot()

    cv2.putText(plotted, str(number), (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    cv2.imshow("Crab detections", plotted)

    # MUST be inside the loop on macOS
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()