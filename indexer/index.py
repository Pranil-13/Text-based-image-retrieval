"""Index fashion images into ChromaDB using CLIP embeddings.

Walks data/images/, extracts 512-dim CLIP embeddings, and upserts them
into a ChromaDB collection for nearest-neighbour retrieval.
Run: python -m indexer.index
"""

from pathlib import Path

import chromadb
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

# ── shared constants ──────────────────────────────────────────────
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMAGE_DIR = "data/images"
CHROMA_DIR = "data/chroma_db"
CHROMA_COLLECTION = "fashion_index"
# ──────────────────────────────────────────────────────────────────

BATCH_SIZE = 32
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_clip(device: torch.device):
    """Load CLIP model and preprocessing transform.

    Returns (model, preprocess, device).
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()
    return model, preprocess


def collect_image_paths(image_dir: str = IMAGE_DIR) -> list[Path]:
    """Return sorted list of image file paths in *image_dir*."""
    d = Path(image_dir)
    if not d.exists():
        raise FileNotFoundError(
            f"Image directory '{image_dir}' not found. "
            "Run `python -m indexer.download_dataset` first."
        )
    paths = sorted(
        p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXT
    )
    if not paths:
        raise FileNotFoundError(f"No images found in '{image_dir}'.")
    return paths


def build_index(
    image_dir: str = IMAGE_DIR,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> int:
    """Encode all images and store embeddings in ChromaDB.

    Returns the number of images indexed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, preprocess = load_clip(device)
    paths = collect_image_paths(image_dir)
    print(f"Found {len(paths)} images in {image_dir}/")

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        # ponytail: default L2 distance is fine; cosine on unit vecs ≡ L2
        metadata={"hnsw:space": "cosine"},
    )

    total_indexed = 0

    for batch_start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Indexing"):
        batch_paths = paths[batch_start : batch_start + BATCH_SIZE]
        tensors, ids, metas = [], [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                ids.append(p.name)
                metas.append({"image_path": f"{image_dir}/{p.name}"})
            except Exception as e:
                print(f"\nSkipping {p.name}: {e}")

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            # unit-normalise
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        emb_list = embeddings.cpu().tolist()

        collection.upsert(ids=ids, embeddings=emb_list, metadatas=metas)
        total_indexed += len(ids)

    print(f"\nDone — indexed {total_indexed} images into '{collection_name}'")
    return total_indexed


if __name__ == "__main__":
    build_index()
