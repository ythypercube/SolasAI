"""
Stack AI Fetcher
- Loads samples from BigCode "The Stack" dataset
- Builds a simple searchable index
- Lets an AI retrieve relevant code examples for coding/Q&A
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class StackAI:
    def __init__(self, subset="python", max_samples=5000):
        print("Loading dataset...")
        self.ds = load_dataset("bigcode/the-stack-smol", data_dir=subset, split="train")
        self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

        print("Embedding model loading...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.texts = []
        self.index = None
        self.embeddings = None

    def build_index(self):
        print("Building embeddings...")

        self.texts = [
            sample["content"][:2000]  # limit size for efficiency
            for sample in self.ds
            if sample.get("content")
        ]

        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print("Index built successfully.")

    def search(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)

        distances, indices = self.index.search(q_emb, k)

        results = []
        for i in indices[0]:
            results.append(self.texts[i])

        return results


# ---------------- Example usage ----------------

if __name__ == "__main__":
    ai = StackAI(subset="python", max_samples=2000)
    ai.build_index()

    while True:
        query = input("\nAsk a coding question: ")
        results = ai.search(query)

        print("\n--- Relevant Stack Snippets ---\n")
        for r in results:
            print(r[:500])
            print("\n----------------------------\n")