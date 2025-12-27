from ultralytics import YOLO
import cv2

# Same as before
model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(source="test_8.png")

# Get the number of boxes and convert the results to a numpy array
number = len(results[0].boxes)
plotted = results[0].plot()
# Add the text to the image
cv2.putText(plotted, str(number), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
# Display the image
cv2.imshow("Crab detections", plotted)

# Wait indefinitely until a key is pressed 
cv2.waitKey(0) 

#Close the window 
cv2.destroyAllWindows()