from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n.pt') 

results = model.train(
    data='dataset_v3/data.yaml',
    epochs=50,
    imgsz=640,
)