import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, dim=384, emb_model=None):
        self.model = emb_model
        self.index = faiss.IndexFlatL2(dim)
        self.job_ids = []

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def build_index(self, job_texts, job_ids=None):
        embeddings = self.encode(job_texts)
        self.index.add(embeddings)
        self.job_ids = job_ids if job_ids else list(range(len(job_texts)))

    def search(self, query, topk=5):
        query_vec = self.encode([query])
        D, I = self.index.search(query_vec, topk)
        results = [(self.job_ids[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results


# Demo
if __name__ == "__main__":
    from ..embedder.embedding_feature import SbertEmbedder, SbertConfig
    retriever = FaissRetriever(emb_model=SbertEmbedder(SbertConfig()))
    jobs = [
        "Data Scientist with Python, SQL, Machine Learning",
        "Frontend Developer ReactJS, CSS, HTML",
        "AI Engineer with NLP, Deep Learning, PyTorch"
    ]
    retriever.build_index(jobs, job_ids=["job1", "job2", "job3"])

    candidate_cv = "I have experience with Python, SQL and Machine Learning"
    print(retriever.search(candidate_cv, topk=2))
