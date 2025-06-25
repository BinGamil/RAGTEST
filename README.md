# WestLaw Semantic Search Project

This repository contains a minimal example of how to build a semantic search system for WestLaw documents. It uses [`sentence-transformers`](https://www.sbert.net/) to create embeddings and [`faiss`](https://github.com/facebookresearch/faiss) to perform nearest-neighbor search.

## Getting Started

1. **Install dependencies** (requires Python 3.8+):
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare your documents**: Export the WestLaw documents you want to search and place them in a directory accessible to the indexing script. Ensure you have the legal right to use these documents.
3. **Index documents**: Modify `westlaw_search/semantic_search.py` to load your documents and add them to the index.
4. **Run a search**:
   ```bash
   python westlaw_search/semantic_search.py
   ```
   The example script demonstrates loading a few documents and querying the index.

## Notes

- This example is intentionally lightweight and meant as a starting point. In a production environment you may want to persist the FAISS index, handle updates, and build a user-facing API.
