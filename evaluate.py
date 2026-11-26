 
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(
    r"runs\detect\runs\railway\yolov8n_baseline\weights\best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("=" * 55)
print("RAILWAY DEFECT DETECTION - TEST EVALUATION")
print("=" * 55)

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data="railway_data.yaml",
    split="test",
    imgsz=416,
    batch=4,
    device="cpu",
    plots=True,
    project="runs/evaluation",
    name="test_results"
)

print("\nFINAL TEST RESULTS")
for key, value in metrics.results_dict.items():
    print(f"{key}: {value}")

print("\nEvaluation completed successfully.")
