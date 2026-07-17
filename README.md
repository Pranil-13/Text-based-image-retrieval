# Multimodal Fashion & Context Retrieval

A CLIP-based image retrieval system that finds fashion and contextual images from natural-language queries. It decomposes complex queries into attribute-level sub-queries (colour, garment, setting) and fuses their similarity scores for stronger compositional matching than vanilla CLIP.

## Architecture

```
Query ──► Decompose ──► Encode sub-queries (CLIP text) ──► Score & fuse
                                                              │
Images ──► Encode (CLIP vision) ──► ChromaDB index ───────────┘
```

| Component | Role |
|---|---|
| **CLIP backbone** | Shared vision-language embedding space (ViT-B/32) |
| **Query decomposition** | Splits a query into attribute sub-queries (e.g. *"bright yellow"*, *"raincoat"*) so each attribute is matched independently |
| **Score fusion** | Averages per-attribute similarities, rewarding images that satisfy *all* aspects of the query |
| **ChromaDB** | Persistent vector store for image embeddings; handles millions of vectors with ANN search |

**Why it beats vanilla CLIP:** A single CLIP embedding for *"A red tie and a white shirt in a formal setting"* blends all attributes into one vector, diluting each. Decomposition keeps them separate, so an image must match colour, garment, *and* setting to rank highly.

## Project Structure

```
glance/
├── data/
│   ├── images/              # Downloaded dataset images
│   └── chroma_db/           # Persisted ChromaDB index
├── indexer/
│   ├── download_dataset.py  # Download & prepare the image dataset
│   └── index.py             # Build CLIP embeddings → ChromaDB
├── retriever/
│   └── search.py            # Decompose, encode, search, fuse
├── demo.py                  # Interactive terminal REPL
├── evaluate.py              # Run evaluation queries & compare
├── requirements.txt
└── README.md
```

## Setup

```bash
# 0. Clone the repository
git clone https://github.com/Pranil-13/Text-based-image-retrieval.git
cd Text-based-image-retrieval

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the image dataset
python -m indexer.download_dataset

# 3. Build the ChromaDB index
python -m indexer.index
```

## Usage

### Interactive demo

```bash
python demo.py
```

Type any natural-language query and get the top-5 matching images with scores. Type `quit` to exit.

### Evaluation

```bash
python evaluate.py
```

Runs the five benchmark queries (see below) through both decomposed search and vanilla CLIP, printing a side-by-side comparison.

### Programmatic

```python
from retriever.search import search

results = search("A person in a bright yellow raincoat.", k=5)
for r in results:
    print(r["image_path"], r["score"])
```

### Adding Custom Images

If you add new images manually to the `data/images/` directory, they won't automatically appear in search results. You must update the vector index to generate their embeddings by re-running the indexer:

```bash
python -m indexer.index
```

## Evaluation Queries

| # | Query |
|---|---|
| 1 | A person in a bright yellow raincoat. |
| 2 | Professional business attire inside a modern office. |
| 3 | Someone wearing a blue shirt sitting on a park bench. |
| 4 | Casual weekend outfit for a city walk. |
| 5 | A red tie and a white shirt in a formal setting. |

## Future Work

- **Domain expansion** — extend retrieval to locations (cities, landmarks) and weather-appropriate outfits.
- **Precision improvements** — fine-tune CLIP on fashion-specific data, apply hard negative mining, and learn attribute weights instead of uniform averaging.
- **Re-ranking** — add a lightweight cross-encoder re-ranker on the top-k candidates.

## Scalability

- **ChromaDB** supports millions of vectors with approximate nearest-neighbour search out of the box.
- **Batch processing** in the indexer encodes images in GPU batches for fast index builds.
- **GPU acceleration** for both indexing and query-time encoding; falls back to CPU automatically.
