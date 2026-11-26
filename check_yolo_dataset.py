from ultralytics.data.utils import check_det_dataset

data = check_det_dataset("railway_data.yaml")

print("\nDATASET LOADED SUCCESSFULLY")
print("Train:", data["train"])
print("Validation:", data["val"])
print("Test:", data.get("test"))
print("Number of classes:", data["nc"])
print("Class names:", data["names"])