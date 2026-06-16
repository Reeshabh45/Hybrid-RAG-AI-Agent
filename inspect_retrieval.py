import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from clear_qdrant import inspect_query

# Load environment variables
load_dotenv()

# Verify paths
sys.path.append(str(Path(__file__).parent))

try:
    from data_loader import embed_chunks
    from vector_db import QdrantStorage
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def format_source(source_path: str) -> str:
    """Format the source path for clean display."""
    try:
        return Path(source_path).name
    except Exception:
        return source_path



def main():
    print("=" * 60)
    print(" RAG Retrieval Quality Inspector")
    print("=" * 60)
    print("Type your search query to inspect retrieval results from Qdrant.")
    print("Type 'exit' or 'quit' to close the inspector.\n")

    store = QdrantStorage()

    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting inspector.")
                break

            top_k_input = input("Enter top_k (default 5): ").strip()
            top_k = 5
            if top_k_input.isdigit():
                top_k = int(top_k_input)

            print(f"\n[Embedding query using HuggingFace...]")
            query_vecs = embed_chunks([query])
            if not query_vecs or len(query_vecs) == 0:
                print("Error: Failed to embed the query.")
                continue
            query_vec = query_vecs[0]

            print(f"[Searching Qdrant for top {top_k} matches...]")
            results = store.search(query_vec, top_k=top_k)

            matches = results.get("matches", [])
            if not matches:
                print("No matches found in the Qdrant database.")
                continue

            print("\n" + "=" * 80)
            print(f"RESULTS FOR: '{query}' (top_k={top_k})")
            print("=" * 80)

            for idx, match in enumerate(matches, 1):
                score_pct = match.score * 100
                source_name = format_source(match.source_id)

                print(f"\n[Match #{idx}] | Score: {match.score:.4f} ({score_pct:.1f}%)")
                print(f"Source: {source_name}")
                print("-" * 80)
                # Print the chunk text. Wrap or format it nicely.
                text_content = match.text.strip()
                print(text_content)
                print("-" * 80)

        except KeyboardInterrupt:
            print("\nExiting inspector.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
