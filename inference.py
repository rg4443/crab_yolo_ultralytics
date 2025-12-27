from ultralytics import YOLO

# Load  trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on a new image
results = model.predict(source="test_8.png")

# Show the detection
results[0].show()
