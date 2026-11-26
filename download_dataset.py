 import kagglehub
import os

print("Starting dataset download...")

path = kagglehub.dataset_download("bibo600/new-images")

print("\nDOWNLOAD COMPLETE")
print("Dataset location:", path)

print("\nFolder structure:")

for root, dirs, files in os.walk(path):
    level = root.replace(path, "").count(os.sep)

    if level > 3:
        dirs[:] = []
        continue

    print("  " * level + os.path.basename(root) + "/")

    for file in files[:10]:
        print("  " * (level + 1) + file)