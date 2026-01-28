from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="dataset_v2/data.yaml", epochs=50, imgsz=640)
