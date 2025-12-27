# import os for reading and writing files
import os

image_folder = "dataset/images/val" # point to our images train or train folder
label_folder = "dataset/labels/val" # point to our labels train or val folder

os.makedirs(label_folder, exist_ok=True)

# Self-explanatory
for filename in os.listdir(image_folder):
    if filename.startswith("crab_original_native_rock_crab") and filename.endswith(".png"):
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_folder, label_filename)
        with open(label_path, "w") as f:
            f.write("2 0.5 0.5 1.0 1.0\n")