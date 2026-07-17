"""Interactive demo for multimodal fashion retrieval.

Prompts the user for natural-language queries and returns the top-k
matching images from the ChromaDB index via decomposed CLIP search.
"""

from retriever.search import search


BANNER = """
╔══════════════════════════════════════════════╗
║   Multimodal Fashion & Context Retrieval     ║
║   Type a query, get matching images.         ║
║   Type 'quit' to exit.                       ║
╚══════════════════════════════════════════════╝
"""


def main():
    """Run the interactive search REPL."""
    print(BANNER)
    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Bye!")
            break
        try:
            results = search(query, k=5)
        except Exception as e:  # ponytail: broad catch — covers missing collection, DB errors
            if "does not exist" in str(e).lower() or "no collection" in str(e).lower():
                print("⚠  ChromaDB collection not found. Run the indexer first:")
                print("   python -m indexer.download_dataset")
                print("   python -m indexer.index")
                continue
            raise
        if not results:
            print("No results found.\n")
            continue
        print(f"\nTop {len(results)} results:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['image_path']}  (score: {r['score']:.4f})")
        print()


if __name__ == "__main__":
    main()
