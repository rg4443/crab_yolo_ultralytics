# Convert images to RBG format for labeling (labelimg requires RBG images)

from pathlib import Path
from PIL import Image

src = Path("./crab")    
dst = Path("images_rgb")
dst.mkdir(exist_ok=True)

for p in src.glob("*"):
    if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}:
        with Image.open(p) as im:
            im = im.convert("RGB")
            out = dst / (p.stem + ".jpg")
            im.save(out, quality=95)
            print("Wrote", out)