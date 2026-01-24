import os

def rename_images(directory_path):
    """
    Renames images starting with specific prefix to 1, 2, 3... 100.
    """
    prefix = "images_to_be_augmented_original"
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    files = [f for f in os.listdir(directory_path) if f.startswith(prefix)]
    files.sort()

    count = 0
    for filename in files:
        count += 1
        if count > 100:
            print("Reached 100 images, stopping.")
            break

        _, extension = os.path.splitext(filename)
        
        new_name = f"{count}{extension}"
        
        old_path = os.path.join(directory_path, filename)
        new_path = os.path.join(directory_path, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except OSError as e:
            print(f"Error renaming {filename}: {e}")

rename_images("./images_to_be_labeled")
