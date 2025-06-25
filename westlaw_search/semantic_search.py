import os
from typing import List

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class WestlawSemanticSearch:
    """Simple semantic search over WestLaw documents."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]):
        """Add documents to the index."""
        embeddings = self.model.encode(docs, convert_to_numpy=True)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 5):
        """Return the top_k most relevant documents for the query."""
        q_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(q_embedding, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results


if __name__ == "__main__":
    # Example usage
    docs = [
        "Example legal document about contract law.",
        "Another document discussing tort law and liabilities.",
    ]
    searcher = WestlawSemanticSearch()
    searcher.add_documents(docs)
    query = "contract liabilities"
    print(searcher.search(query))
