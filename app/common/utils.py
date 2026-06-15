"""

"""
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import cv2

MAX_LONG_SIDE: int = 800    # Resize longest dimension to this (px). Reduces RAM ~2-4x.
                             # Safe floor for PaddleOCR legibility: 640. Max recommended: 1280.
MIN_LONG_SIDE: int = 640    # Never downsample below this (avoids blurring small text).
CONF_THRESHOLD: float = 0.5 # Discard detections below this confidence score.

class InvoiceModel(BaseModel):
    invoice_number: str = Field()
    invoice_date: str = Field()
    invoice_items: list = Field()

def _preprocess(image_path: Optional[str | Path | Image.Image | bytes]) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load and preprocess an image for PaddleOCR:
      1. EXIF-aware load  → corrects phone/camera auto-rotation (portrait vs landscape).
      2. RGB conversion   → strips alpha channel, normalises colour space.
      3. Smart resize     → shrinks high-res inputs to MAX_LONG_SIDE on the longest axis,
                            keeping aspect ratio. Never upscales small images.
                            Uses INTER_AREA (best quality for downsampling).

    Returns:
        img_rgb  : preprocessed uint8 NumPy array (H, W, 3) ready for PaddleOCR.
        orig_hw  : (H, W) of the original image before resizing.
    """
    # --- 1. Load & honour EXIF rotation ---
    if isinstance(image_path, bytes):
        pil_img = Image.open(BytesIO(image_path))
    elif not isinstance(image_path, Image.Image):
        pil_img = Image.open(image_path)
    else:
        pil_img = image_path.copy()
    pil_img = ImageOps.exif_transpose(pil_img)  # Rotates in-memory; no quality loss

    # --- 2. Ensure RGB (drop alpha, handle palette/grayscale) ---
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    orig_hw = img_rgb.shape[:2]   # (H, W)
    h, w = orig_hw

    # --- 3. Resize: longest side → MAX_LONG_SIDE (only downscale) ---
    long_side = max(h, w)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        new_h = max(int(h * scale), MIN_LONG_SIDE if h >= w else 1)
        new_w = max(int(w * scale), MIN_LONG_SIDE if w > h else 1)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(
            f"Resized {w}x{h} → {new_w}x{new_h}",
        )
    else:
        print(f"Image {w}x{h} is within limits — no resize needed.")

    return img_rgb, orig_hw