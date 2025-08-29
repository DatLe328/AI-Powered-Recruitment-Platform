from typing import Iterable
import numpy as np
from ml.embeddings import embed_texts
from ml.vectorstore import faiss_store

def ensure_faiss_loaded() -> bool:
    return faiss_store.load()

def add_one_to_faiss(cv_id: int | str, resume_text: str):
    emb = embed_texts([resume_text])[0]  # np.ndarray (D,)
    if not faiss_store.is_loaded():
        faiss_store.build_new(np.asarray([emb]), [cv_id], kind="hnsw")
    else:
        faiss_store.add(np.asarray([emb]), [cv_id])

def add_many_to_faiss(items: Iterable[tuple[int | str, str]]):
    ids, texts = zip(*items)
    embs = embed_texts(list(texts))
    if not faiss_store.is_loaded():
        faiss_store.build_new(np.asarray(embs), list(ids), kind="hnsw")
    else:
        faiss_store.add(np.asarray(embs), list(ids))