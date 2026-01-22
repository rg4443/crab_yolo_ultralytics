import Augmentor

p = Augmentor.Pipeline("images_to_be_augmented")

# Geometric transformations
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.3)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)

# Lighting and color
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)

p.sample(100)  # create 100 variations
