
from ultralytics import YOLO
import torch

print("=" * 50)
print("RAILWAY TRACK DEFECT DETECTION")
print("=" * 50)

print("Device: CPU")
print("PyTorch:", torch.__version__)

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="railway_data.yaml",
    epochs=10,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=0,
    patience=5,
    project="runs/railway",
    name="yolov8n_baseline",
    pretrained=True,
    plots=True,
    save=True,
    verbose=True
)

print("\nTRAINING COMPLETED")
print("Best model: runs/railway/yolov8n_baseline/weights/best.pt")
print("Last model: runs/railway/yolov8n_baseline/weights/last.pt")
