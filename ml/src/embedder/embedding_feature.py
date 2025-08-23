from __future__ import annotations
import os, re, hashlib, pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

class EmbCache:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._mem: Dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._mem = pickle.load(f)
            except Exception:
                self._mem = {}

    def _key(self, model: str, text: str) -> str:
        return hashlib.sha1((model + "|" + _norm_text(text)).encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        if not self.path: return None
        return self._mem.get(self._key(model, text))

    def put(self, model: str, text: str, vec: np.ndarray):
        if not self.path: return
        self._mem[self._key(model, text)] = vec

    def save(self):
        if not self.path: return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._mem, f)

@dataclass
class SbertConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 64
    normalize: bool = True
    cache_path: Optional[str] = None

class SbertEmbedder:
    def __init__(self, cfg: SbertConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.batch_size = int(cfg.batch_size)
        self.normalize = bool(cfg.normalize)
        self.cache = EmbCache(cfg.cache_path)

    @staticmethod
    def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return x / nrm

    def encode(self, texts: List[str]) -> np.ndarray:
        outs: List[Optional[np.ndarray]] = [None] * len(texts)
        todo_idx, todo_txt = [], []
        for i, t in enumerate(texts):
            v = self.cache.get(self.cfg.model_name, t) if self.cache.path else None
            if v is not None:
                outs[i] = v
            else:
                todo_idx.append(i); todo_txt.append(t)

        if todo_txt:
            vecs = self.model.encode(
                todo_txt, batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=False
            ).astype(np.float32)
            for i, v in zip(todo_idx, vecs):
                outs[i] = v
                if self.cache.path: self.cache.put(self.cfg.model_name, texts[i], v)
            if self.cache.path: self.cache.save()

        arr = np.stack(outs, axis=0).astype(np.float32)
        if self.normalize:
            arr = self._l2norm(arr)
        return arr

def add_sbert_similarity_feature(
    df,
    embedder: SbertEmbedder,
    jd_col: str = "job_description_text",
    cv_col: str = "resume_text",
    out_col: str = "emb_cosine",
    per_jd_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    jd_tbl = df[["jd_id", jd_col]].drop_duplicates("jd_id")
    cv_tbl = df[["cv_id", cv_col]].drop_duplicates("cv_id")

    E_jd = embedder.encode(jd_tbl[jd_col].astype(str).tolist())
    E_cv = embedder.encode(cv_tbl[cv_col].astype(str).tolist())

    jd2row = {j: i for i, j in enumerate(jd_tbl["jd_id"].tolist())}
    cv2row = {c: i for i, c in enumerate(cv_tbl["cv_id"].tolist())}

    sims = np.empty(len(df), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        j = jd2row[getattr(r, "jd_id")]
        c = cv2row[getattr(r, "cv_id")]
        sims[i] = float((E_jd[j] * E_cv[c]).sum())

    df[out_col] = sims
    if per_jd_norm:
        df[out_col + "_norm"] = df.groupby("jd_id")[out_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
    return E_jd, E_cv
