from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on a new image
results = model.predict(source="european_green_crab.jpg")

# Show the detection
results[0].show()
