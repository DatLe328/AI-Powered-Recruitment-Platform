import os, json, threading, time
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError as e:
    raise RuntimeError("Cần cài faiss-cpu: pip install faiss-cpu") from e

_LOCK = threading.RLock()
_INDEX = None
_IDMAP = []     # list các cv_id cùng thứ tự vector trong index
_DIM = None

def _base_dir() -> Path:
    # Lấy từ settings nếu có
    try:
        from django.conf import settings
        p = Path(getattr(settings, "FAISS_INDEX_DIR", Path("ml/models/faiss")))
    except Exception:
        p = Path("ml/models/faiss")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _paths():
    base = _base_dir()
    return {
        "index": base / "index.faiss",
        "ids": base / "ids.jsonl",
        "meta": base / "meta.json"
    }

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _create_index(dim: int, kind: str = "hnsw", M: int = 32, efConstruction: int = 200):
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = 64
    else:
        raise ValueError("Unsupported index kind")
    return index

def build_new(embeddings: np.ndarray, ids, kind: str = "hnsw", meta: dict | None = None):
    """Xây index FAISS từ đầu và persist ra file; cập nhật cache RAM."""
    global _INDEX, _IDMAP, _DIM
    if len(embeddings) != len(ids):
        raise ValueError("embeddings và ids phải cùng độ dài")
    if len(ids) == 0:
        raise ValueError("Không có dữ liệu để build index")

    embs = _normalize(np.asarray(embeddings))
    dim = embs.shape[1]
    p = _paths()

    with _LOCK:
        index = _create_index(dim, kind=kind)
        index.add(embs)
        faiss.write_index(index, str(p["index"]))

        with open(p["ids"], "w", encoding="utf-8") as f:
            for _id in ids:
                f.write(json.dumps(_id, ensure_ascii=False) + "\n")

        meta = (meta or {}) | {
            "dim": dim, "count": len(ids),
            "metric": "ip_cosine", "index_kind": kind,
            "created_at": int(time.time())
        }
        p["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        _INDEX = index
        _IDMAP = list(ids)
        _DIM = dim

def load() -> bool:
    """Load index + idmap vào RAM từ file. Trả về True nếu có file."""
    global _INDEX, _IDMAP, _DIM
    p = _paths()
    if not p["index"].exists() or not p["ids"].exists():
        return False
    with _LOCK:
        index = faiss.read_index(str(p["index"]))
        ids = []
        with open(p["ids"], "r", encoding="utf-8") as f:
            for line in f:
                ids.append(json.loads(line))
        _INDEX = index
        _IDMAP = ids
        _DIM = _INDEX.d
    return True

def is_loaded() -> bool:
    return _INDEX is not None

def add(embeddings: np.ndarray, ids):
    """Append vector + id rồi persist; yêu cầu đã load()."""
    assert is_loaded(), "Index chưa load"
    if len(embeddings) != len(ids):
        raise ValueError("embeddings và ids phải cùng độ dài")
    embs = _normalize(np.asarray(embeddings))
    p = _paths()
    with _LOCK:
        _INDEX.add(embs)
        with open(p["ids"], "a", encoding="utf-8") as f:
            for _id in ids:
                f.write(json.dumps(_id, ensure_ascii=False) + "\n")
        faiss.write_index(_INDEX, str(p["index"]))
        _IDMAP.extend(ids)

def search(query_embeddings: np.ndarray, topk: int = 10):
    """Trả về danh sách kết quả cho từng query: [[(cv_id, score), ...], ...]."""
    assert is_loaded(), "Index chưa load"
    q = _normalize(np.asarray(query_embeddings))
    with _LOCK:
        scores, idxs = _INDEX.search(q, topk)
    results = []
    for row_scores, row_idxs in zip(scores, idxs):
        row = []
        for s, i in zip(row_scores, row_idxs):
            if i == -1:
                continue
            row.append((_IDMAP[i], float(s)))
        results.append(row)
    return results
