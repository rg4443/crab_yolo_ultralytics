import Augmentor

p = Augmentor.Pipeline("images_to_be_augmented")

# --- 1. Stronger Geometric Deformations (Forces shape invariance) ---
# Increased rotation range slightly
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)

# Added Skew: This simulates looking at the object from different angles
p.skew(probability=0.5, magnitude=0.5) 

# Added Distortion: This warps the image like looking through wavy glass.
# It is excellent for preventing the model from memorizing exact pixel edges.
p.random_distortion(probability=0.6, grid_width=4, grid_height=4, magnitude=8)

# --- 2. Occlusion (The most important for your specific problem) ---
# This forces the model to recognize an object even if part of it is hidden.
# (Simulates "Cutout" augmentation)
p.random_erasing(probability=0.5, rectangle_area=0.2)

# --- 3. Tougher Lighting & Noise ---
# Widened the brightness/contrast gap to simulate very dark or overexposed scenes
p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

# Added Color Jitter (helps if color isn't the defining feature of your object)
p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)

# --- 4. Standard Flips ---
p.flip_left_right(probability=0.5)
# Be careful with top_bottom flip if your objects never appear upside down in real life.
p.flip_top_bottom(probability=0.3)

p.sample(100)