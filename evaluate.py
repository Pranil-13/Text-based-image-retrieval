"""Evaluation script for multimodal fashion retrieval.

Runs five predefined queries through both the decomposed search pipeline
and a vanilla CLIP baseline, printing side-by-side results for comparison.
"""

import numpy as np
import open_clip
import chromadb
import torch

from retriever.search import (
    search,
    decompose_query,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
)

QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

DIVIDER = "=" * 60


def vanilla_clip_search(query: str, model, tokenizer, collection, k: int = 5) -> list[dict]:
    """Search ChromaDB with the raw query embedding (no decomposition).

    This is the baseline: encode the full query with the same CLIP prompt
    template and retrieve nearest neighbours.
    """
    prompt = f"a photo of {query.lower().strip('.')}"
    tokens = tokenizer([prompt])
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    emb_np = emb.cpu().numpy()
    results = collection.query(
        query_embeddings=emb_np.tolist(),
        n_results=k,
        include=["metadatas", "embeddings"],
    )

    # Compute cosine similarity (dot product on unit vectors)
    result_embs = np.array(results["embeddings"][0])
    norms = np.linalg.norm(result_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    result_embs = result_embs / norms
    scores = result_embs @ emb_np.flatten()

    out = []
    for img_id, score, meta in zip(results["ids"][0], scores, results["metadatas"][0]):
        out.append({"image_path": meta.get("image_path", ""), "score": float(score), "id": img_id})
    return out


def print_results(label: str, results: list[dict]):
    """Pretty-print a ranked result list."""
    print(f"  [{label}]")
    if not results:
        print("    (no results)")
        return
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r['image_path']}  (score: {r['score']:.4f})")


def run_evaluation():
    """Run all evaluation queries and print a summary."""
    print(DIVIDER)
    print("  Multimodal Fashion Retrieval - Evaluation")
    print(DIVIDER)

    # Load model and collection once for vanilla baseline
    model, _, _preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION)

    summary = []

    for idx, query in enumerate(QUERIES, 1):
        print(f"\nQuery {idx}: \"{query}\"")
        print("-" * 50)

        sub_queries = decompose_query(query)
        print(f"  Sub-queries: {sub_queries}")

        decomposed_results = search(query, k=5)
        vanilla_results = vanilla_clip_search(query, model, tokenizer, collection, k=5)

        print()
        print_results("Decomposed Search", decomposed_results)
        print()
        print_results("Vanilla CLIP", vanilla_results)

        # Compare rankings — how many unique images does decomposition surface?
        dec_ids = {r["id"] for r in decomposed_results}
        van_ids = {r["id"] for r in vanilla_results}
        unique_to_decomposed = dec_ids - van_ids
        if unique_to_decomposed:
            print(f"\n  New images surfaced by decomposition: {unique_to_decomposed}")
        print()

        top_decomposed = decomposed_results[0]["score"] if decomposed_results else 0.0
        top_vanilla = vanilla_results[0]["score"] if vanilla_results else 0.0
        summary.append({
            "query": query,
            "decomposed_ids": [r["id"] for r in decomposed_results],
            "vanilla_ids": [r["id"] for r in vanilla_results],
            "top_decomposed": top_decomposed,
            "top_vanilla": top_vanilla,
        })
        print(DIVIDER)

    # --- Summary ---
    print("\n" + DIVIDER)
    print("  SUMMARY")
    print(DIVIDER)
    print(f"  {'#':<4} {'Decomposed Top-1':>18} {'Vanilla Top-1':>15}  Query")
    print(f"  {'-'*4} {'-'*18} {'-'*15}  {'-'*35}")
    for i, s in enumerate(summary, 1):
        # Check if rankings differ (decomposition surfaced different images)
        different = s["decomposed_ids"] != s["vanilla_ids"]
        marker = "*" if different else "="
        print(
            f"  {i:<4} {s['top_decomposed']:>18.4f} {s['top_vanilla']:>15.4f}  {s['query'][:35]}  {marker}"
        )
    diffs = sum(1 for s in summary if s["decomposed_ids"] != s["vanilla_ids"])
    print(f"\n  Rankings differ on {diffs}/{len(summary)} queries (* = different ranking).")
    print("  Note: raw scores are not directly comparable between methods.")
    print("  Decomposed search targets better attribute coverage, not higher raw scores.")
    print(DIVIDER)


if __name__ == "__main__":
    run_evaluation()
