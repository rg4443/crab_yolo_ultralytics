from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt') 

results = model.train(
    data='v4/data.yaml',
    epochs=50,
    imgsz=640,
)   