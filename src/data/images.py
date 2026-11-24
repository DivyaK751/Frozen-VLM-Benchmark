"""Image loading and SigLIP preprocessing (D4).

Decisions:
- Images are stretched to 384x384 (SigLIP's default), not padded to square.
- Non-RGB images (grayscale, palette, RGBA, CMYK) are converted to RGB, like LLaVA.
- The slow (PIL) processor is used explicitly so train and eval always match.
"""

from functools import lru_cache

from PIL import Image

ENCODER_NAME = "google/siglip2-so400m-patch14-384"
IMAGE_SIZE = 384


def load_image(path):
    """Open an image and return it as RGB.

    Returns (image, None) on success, or (None, error_message) if the file
    is missing, truncated or not a valid image.
    """
    try:
        with Image.open(path) as im:
            im.load()                      # read the pixels now, so a broken file fails here
            if im.mode != "RGB":           # grayscale (L), palette (P), RGBA, CMYK, ...
                im = im.convert("RGB")     # same as LLaVA: transparency is dropped
            else:
                im = im.copy()             # detach from the closed file
        return im, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@lru_cache(maxsize=1)
def get_processor(name=ENCODER_NAME):
    """Load the SigLIP image processor once and reuse it."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(name, use_fast=False)


def preprocess(images):
    """Turn a list of RGB PIL images into a float tensor [N, 3, 384, 384] in [-1, 1]."""
    return get_processor()(images=images, return_tensors="pt")["pixel_values"]


def blank_image(size=IMAGE_SIZE):
    """Gray image for the blank-image test: after normalization every pixel is ~0
    (the SigLIP mean of 0.5 maps to 0)."""
    return Image.new("RGB", (size, size), (128, 128, 128))
