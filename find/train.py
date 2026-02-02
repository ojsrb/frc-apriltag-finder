from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(
    data="yolo/dataset/data.yaml",
    epochs=32,
    imgsz=640,
    device="mps",
    save_period=1,
)

model.save("find/apriltags.pt")