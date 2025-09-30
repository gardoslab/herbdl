# image_utils.py
import os
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_file_size_in_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

def resize_with_aspect_ratio(
    image_path,
    output_path,
    max_size=1600,
    format="JPEG",
    quality=85
):
    """
    Downscale to 'max_size' longest edge while preserving aspect ratio.
    Convert to JPEG (RGB) and strip alpha safely.
    Returns (changed: bool, final_size: (w, h))
    """
    with Image.open(image_path) as img:
        orig_format = (img.format or "").upper()
        orig_size = img.size

        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            alpha = img.split()[-1] if img.mode != "RGB" else None
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if alpha is not None:
                bg.paste(img, mask=alpha)
            else:
                bg.paste(img)
            img = bg
        else:
            img = img.convert("RGB")

        w, h = orig_size
        longest = max(w, h)
        scale = min(max_size / float(longest), 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        else:
            new_size = (w, h)

        img.save(output_path, format=format, optimize=True, progressive=True, quality=quality)

        changed = (new_size != orig_size) or (orig_format != format)
        return changed, new_size
