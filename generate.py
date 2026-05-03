import albumentations as A
import cv2
import os
import glob

# Configuration 
FOLDERS = ['train', 'val'] # Have train and val folders be in the root of the project.
MULTIPLIERS = {'train': 2, 'val': 1} # For each image create x augmentations. (DO NOT CHANGE THE VAL)
DATASET_VERSION_NUM = 6 # Current Dataset iteration.

import albumentations as A

transform = A.Compose([
    A.SafeRotate(limit=180, p=0.8), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.RandomResizedCrop(size=(640, 640), scale=(0.4, 1.0), p=0.5),

    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.4),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3), 

    A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3), 

    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3)
    
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

for split in FOLDERS:
    input_dir = f"{split}"
    label_dir = f"{split}"
    output_img_dir = f"dataset_v{7}/images/{split}"
    output_lbl_dir = f"dataset_v{7}/labels/{split}"
    
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
        count = MULTIPLIERS[split]
        for i in range(count):
            try:
                if split == 'train':
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    final_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    final_bboxes = augmented['bboxes']
                    final_class_labels = augmented['class_labels']
                    save_name = f"aug_{filename_only}{i}"
                else:
                    # For 'val', use the original image and labels as-is
                    final_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    final_bboxes = bboxes
                    final_class_labels = class_labels
                    save_name = f"{filename_only}" # Keep original name for val

                cv2.imwrite(f"{output_img_dir}/{save_name}.jpg", final_img)
                
                # Save Label
                with open(f"{output_lbl_dir}/{save_name}.txt", 'w') as f:
                    for box, cls in zip(augmented['bboxes'], augmented['class_labels']):
                        # Clamp again just to be safe for yolo
                        x = min(max(box[0], 0), 1)
                        y = min(max(box[1], 0), 1)
                        wd = min(max(box[2], 0), 1)
                        ht = min(max(box[3], 0), 1)
                        f.write(f"{cls} {x:.6f} {y:.6f} {wd:.6f} {ht:.6f}\n")
                        
            except Exception as e:
                print(f"Skipped {save_name}: {e}")

print("Done.")