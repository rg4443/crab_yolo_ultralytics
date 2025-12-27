from ultralytics import YOLO

model = YOLO("yolo11n.pt") # use a pre-trained model to train on 
# Point to our .yaml configuration file, with 50 epochs, and resizing teh image to 640x640 pixels.
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)

