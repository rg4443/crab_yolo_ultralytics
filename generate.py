import albumentations as A
import cv2
import os
import glob
import random

folders = ['train', 'val']
multipliers = {'train': 15, 'val': 8} 

transform = A.Compose([
    A.Rotate(limit=20, p=1.0),                  # Rotation
    A.RandomBrightnessContrast(p=1.0),          # Lighting
    A.HorizontalFlip(p=0.5),                    # Flip half the time
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),# Grain
    A.Blur(blur_limit=3, p=0.3),                # Slight blur
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5) # Color shift
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

for split in folders:
    input_dir = f"data/{split}_parents"
    label_dir = f"data/{split}_labels"
    output_img_dir = f"data/{split}_parents"
    output_lbl_dir = f"data/{split}_labels"
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    print(f"Processing {split} set: Found {len(image_files)} parent images.")

    for img_file in image_files:
        filename_only, file_extension = os.path.splitext(os.path.basename(img_file))

        # Read Image
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Read Label
        label_file = os.path.join(label_dir, filename_only + '.txt')
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    # Fix potential float errors by clamping to 0-1
                    x, y, wd, ht = [float(p) for p in parts[1:]]
                    bboxes.append([x, y, wd, ht])
                    class_labels.append(cls)
        
        # Generate Variations
        count = multipliers[split]
        for i in range(count):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save Image
                save_name = f"aug_{filename_only}{i}{file_extension}"
                
                final_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{output_img_dir}/{save_name}.jpg", final_img)
                
                # Save Label
                with open(f"{output_lbl_dir}/{save_name}.txt", 'w') as f:
                    for box, cls in zip(augmented['bboxes'], augmented['class_labels']):
                        # Clamp again just to be safe for YOLO
                        x = min(max(box[0], 0), 1)
                        y = min(max(box[1], 0), 1)
                        wd = min(max(box[2], 0), 1)
                        ht = min(max(box[3], 0), 1)
                        f.write(f"{cls} {x:.6f} {y:.6f} {wd:.6f} {ht:.6f}\n")
                        
            except Exception as e:
                print(f"Skipped {save_name}: {e}")

print("Done.")