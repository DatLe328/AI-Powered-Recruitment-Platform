# ml/embeddings.py
import numpy as np

# Ưu tiên dùng embedder có sẵn của bạn, nếu không có thì fallback SBERT
try:
    from ml.apis import embed_texts as _embed_texts  # nếu repo bạn đã có
    def embed_texts(texts: list[str]) -> np.ndarray:
        arr = _embed_texts(texts)
        arr = np.asarray(arr, dtype="float32")
        return arr
except Exception:
    # Fallback: SBERT đa ngôn ngữ nhẹ, đủ tốt cho CV/JD
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_texts(texts: list[str]) -> np.ndarray:
        arr = _SBERT.encode(texts, batch_size=64, show_progress_bar=False)
        return np.asarray(arr, dtype="float32")
