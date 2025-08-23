from __future__ import annotations
import os, json, threading, hashlib, re
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
import time

from .src.models import CVJDXGBRanker

# ============== internal state ==============
_LOCK = threading.Lock()
_MODEL: CVJDXGBRanker | None = None
_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE / "models"
_DEFAULT_MODEL_PATH  = _MODELS_DIR / "xgb_ranker_v1.json"
_DEFAULT_CONFIG_PATH = _MODELS_DIR / "xgb_ranker_config_v1.json"

# ============== helpers ==============
def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _stable_hash_id(text: str, prefix: str) -> str:
    h = hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}"

def _merge_jd(job_requirement: Optional[str], job_description: Optional[str]) -> str:
    req  = (job_requirement or "").strip()
    desc = (job_description or "").strip()
    if req and desc:
        return req + "\n\n" + desc
    return req or desc

# ============== model lifecycle ==============
def load_model(force: bool = False,
               model_path: Optional[str | Path] = None,
               config_path: Optional[str | Path] = None,
               emb_device: Optional[str] = None) -> CVJDXGBRanker:
    global _MODEL
    if _MODEL is not None and not force:
        return _MODEL

    with _LOCK:
        if _MODEL is not None and not force:
            return _MODEL

        mp = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        cp = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not mp.exists():
            raise FileNotFoundError(f"Model file not found: {mp.resolve()}")
        if not cp.exists():
            raise FileNotFoundError(f"Config file not found: {cp.resolve()}")

        model = CVJDXGBRanker.load(str(mp), str(cp))

        # SBERT device (nếu make_features có dùng embedding)
        if emb_device and hasattr(model, "feat_cfg"):
            try:
                model.feat_cfg.emb_device = emb_device
            except Exception:
                pass

        _MODEL = model
        return _MODEL

def reload_model() -> CVJDXGBRanker:
    return load_model(force=True)

# ============== main API ==============
def rank_cv_for_jd(
    *,
    job_requirement: Optional[str],
    job_description: Optional[str],
    candidates: List[Dict[str, Any]],
    topk: Optional[int] = None,
) -> List[Dict[str, Any]]:
    init_time = time.time()
    model = load_model()

    jd_text = _merge_jd(job_requirement, job_description)
    jd_id = _stable_hash_id(jd_text, "jd-")

    rows = []
    for i, c in enumerate(candidates, 1):
        cv_text = c.get("resume_text", "")
        cv_id = c.get("cv_id") or _stable_hash_id(cv_text, "cv-")
        rows.append({
            "jd_id": jd_id,
            "job_description_text": jd_text,
            "cv_id": cv_id,
            "resume_text": cv_text
        })
    df_pairs = pd.DataFrame(rows)

    ranked = model.rank(df_pairs, topk=topk).copy() 
    ranked["score"] = ranked["pred"].astype(float)

    ranked = ranked.sort_values(["jd_id", "score"], ascending=[True, False]).reset_index(drop=True)
    print(f"[Ranking] completed in {time.time() - init_time:.2f} seconds")
    return [
        {"jd_id": r.jd_id, "cv_id": r.cv_id, "pred": float(r.pred), "score": float(r.score)}
        for r in ranked.itertuples(index=False)
    ]
