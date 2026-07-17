"""Attribute-aware fashion retriever.

Decomposes natural-language queries into color/garment/setting sub-queries,
retrieves candidates from multiple embedding perspectives, and fuses scores
for better compositional retrieval than vanilla CLIP similarity.
"""

import re

import numpy as np
import open_clip
import chromadb
import torch

# ── shared constants ────────────────────────────────────────────────────────
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CHROMA_PERSIST_DIR = "data/chroma_db"
CHROMA_COLLECTION = "fashion_index"

# ── attribute vocabularies ──────────────────────────────────────────────────
COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "pink", "orange",
    "purple", "brown", "gray", "grey", "beige", "navy", "maroon", "bright",
    "dark", "light", "crimson", "teal", "ivory",
]

GARMENTS = [
    "shirt", "t-shirt", "tshirt", "blazer", "jacket", "coat", "raincoat",
    "pants", "jeans", "trousers", "skirt", "dress", "suit", "tie",
    "hoodie", "sweater", "shorts", "blouse", "vest", "scarf", "hat",
    "cap", "shoes", "boots", "sneakers", "heels", "sandals",
    "button-down", "outerwear", "attire", "outfit", "wear",
]

SETTINGS = [
    "office", "street", "park", "home", "indoor", "outdoor", "urban",
    "city", "formal", "casual", "beach", "garden", "cafe", "restaurant",
    "gym", "studio", "mall", "bench", "building", "modern", "professional",
    "business", "weekend",
]

# ponytail: generic garments that are too vague without a color modifier
_GENERIC_GARMENTS = {"attire", "outfit", "wear", "outerwear"}
_COLOR_SET = set(COLORS)
_SETTING_SET = set(SETTINGS)


def decompose_query(query: str) -> list[str]:
    """Break a query into attribute-specific sub-queries.

    Strategy:
      1. Find garment words; look 1-3 words before each for a color modifier.
         → "a photo of a {color} {garment}" or "a photo of a person wearing a {garment}"
         Generic garments without color are skipped (too vague).
      2. Find setting/environment words.
         → "a photo taken in a {setting} environment"
      3. Fall back to [query] if nothing was extracted.
    """
    words = [re.sub(r"[^\w\-]", "", w) for w in query.lower().split()]
    sub_queries: list[str] = []

    # ── garment + optional color ────────────────────────────────────────
    for i, w in enumerate(words):
        if w not in GARMENTS:
            continue
        # look at up to 3 preceding words for a color
        color_found = None
        for offset in range(1, min(4, i + 1)):
            if words[i - offset] in _COLOR_SET:
                color_found = words[i - offset]
                break
        if color_found:
            sub_queries.append(f"a photo of a {color_found} {w}")
        elif w not in _GENERIC_GARMENTS:
            # only keep specific garments, skip vague ones
            sub_queries.append(f"a photo of a person wearing a {w}")

    # ── setting / environment ───────────────────────────────────────────
    for w in words:
        if w in _SETTING_SET:
            sub_queries.append(f"a photo taken in a {w} environment")

    return sub_queries if sub_queries else [query]


def search(query: str, k: int = 5) -> list[dict]:
    """Return top-k fashion images matching *query*.

    Improvements over vanilla CLIP:
      1. Query decomposition into attribute sub-queries
      2. Multi-query candidate retrieval (not just full query)
      3. Adaptive score fusion based on sub-query specificity

    Returns list of ``{"image_path": str, "score": float, "id": str}``.
    """
    # ── load CLIP ───────────────────────────────────────────────────────
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    # ── load ChromaDB collection ────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    # ── encode queries ──────────────────────────────────────────────────
    sub_queries = decompose_query(query)

    # Use CLIP prompt template for the full query too
    full_prompt = f"a photo of {query.lower().strip('.')}"
    all_texts = [full_prompt] + sub_queries  # index 0 = full query
    tokens = tokenizer(all_texts)

    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    full_emb = text_feats[0].cpu().numpy()   # (D,)
    sub_embs = text_feats[1:].cpu().numpy()  # (S, D)

    # ── multi-query candidate retrieval ─────────────────────────────────
    # ponytail: pull candidates from EACH query perspective, then merge.
    # This finds images that match individual attributes even if they
    # don't rank high for the combined query. Upgrade: use ANN pre-filter.
    n_per_query = k * 2
    candidate_ids = set()

    # Query with full embedding
    results = collection.query(
        query_embeddings=[full_emb.tolist()],
        n_results=n_per_query,
        include=["embeddings", "metadatas"],
    )
    candidate_ids.update(results["ids"][0])

    # Query with each sub-query embedding
    for sub_emb in sub_embs:
        sub_results = collection.query(
            query_embeddings=[sub_emb.tolist()],
            n_results=n_per_query,
            include=["embeddings", "metadatas"],
        )
        candidate_ids.update(sub_results["ids"][0])

    # ── fetch all candidate embeddings ──────────────────────────────────
    candidate_ids = list(candidate_ids)
    fetched = collection.get(
        ids=candidate_ids,
        include=["embeddings", "metadatas"],
    )

    ids = fetched["ids"]
    embeddings = np.array(fetched["embeddings"])    # (C, D)
    metadatas = fetched["metadatas"]

    # normalise stored embeddings (should already be, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    # ── adaptive score fusion ───────────────────────────────────────────
    base_scores = embeddings @ full_emb                    # (C,)

    if len(sub_embs) > 0:
        attr_scores = embeddings @ sub_embs.T              # (C, S)

        # Determine how specific our sub-queries are
        has_color_garment = any(
            any(c in sq for c in COLORS) and any(g in sq for g in GARMENTS)
            for sq in sub_queries
        )

        if has_color_garment:
            # Good decomposition — weight attributes higher
            # Use max over attribute scores to reward images matching ALL attributes
            # ponytail: geometric mean rewards images that satisfy every attribute
            attr_min = attr_scores.min(axis=1)   # worst attribute match
            attr_mean = attr_scores.mean(axis=1)
            # Blend: penalise images missing any attribute
            attr_combined = 0.6 * attr_mean + 0.4 * attr_min
            alpha = 0.35  # base weight
        else:
            # Weak decomposition — lean more on base score
            attr_combined = attr_scores.mean(axis=1)
            alpha = 0.6   # base weight

        final_scores = alpha * base_scores + (1 - alpha) * attr_combined
    else:
        final_scores = base_scores

    # ── top-k ───────────────────────────────────────────────────────────
    top_idx = np.argsort(final_scores)[::-1][:k]

    return [
        {
            "image_path": metadatas[i].get("image_path", ""),
            "score": float(final_scores[i]),
            "id": ids[i],
        }
        for i in top_idx
    ]


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = input("Query: ").strip()
    if not q:
        raise SystemExit("empty query")
    hits = search(q)
    print(f"\nSub-queries: {decompose_query(q)}\n")
    for rank, h in enumerate(hits, 1):
        print(f"  {rank}. [{h['score']:.4f}] {h['image_path']}  (id={h['id']})")
