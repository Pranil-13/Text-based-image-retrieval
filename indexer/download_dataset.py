"""Download ~1000 fashion images from HuggingFace Fashionpedia dataset.

Saves images to data/images/ as img_0001.jpg, img_0002.jpg, etc.
Run: python -m indexer.download_dataset
"""

import os
from pathlib import Path

from tqdm import tqdm

# ── shared constants ──────────────────────────────────────────────
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMAGE_DIR = "data/images"
CHROMA_DIR = "data/chroma_db"
CHROMA_COLLECTION = "fashion_index"
# ──────────────────────────────────────────────────────────────────

MAX_IMAGES = 1000
HF_DATASET = "detection-datasets/fashionpedia"


def download_images(out_dir: str = IMAGE_DIR, limit: int = MAX_IMAGES) -> int:
    """Download fashion images from HuggingFace and save as JPEGs.

    Returns the number of images successfully saved.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets")
        return 0

    try:
        print(f"Loading dataset '{HF_DATASET}' (train split)...")
        ds = load_dataset(HF_DATASET, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("\nManual download instructions:")
        print(f"  1. Go to https://huggingface.co/datasets/{HF_DATASET}")
        print(f"  2. Download images and place them in '{out_dir}/'")
        print("  3. Re-run the indexer with: python -m indexer.index")
        return 0

    n = min(limit, len(ds))
    saved = 0

    for i in tqdm(range(n), desc="Saving images"):
        try:
            img = ds[i]["image"]
            # ponytail: always convert to RGB — some PNGs have alpha
            img = img.convert("RGB")
            fname = f"img_{i + 1:04d}.jpg"
            img.save(out / fname, "JPEG", quality=90)
            saved += 1
        except Exception as e:
            print(f"\nSkipping image {i}: {e}")

    print(f"\nDone — saved {saved}/{n} images to {out_dir}/")
    return saved


if __name__ == "__main__":
    download_images()
