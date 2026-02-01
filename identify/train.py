from ultralytics import YOLO

model = YOLO("yolov8n-cls.yaml")

results = model.train(data="yolo/dataset", epochs=32, imgsz=64, device="mps")

model.save("yolo/identify.pt")