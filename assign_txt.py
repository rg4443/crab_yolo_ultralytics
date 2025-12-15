import os

image_folder = "dataset/images/val"
label_folder = "dataset/labels/val"

os.makedirs(label_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_folder, label_filename)
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")
