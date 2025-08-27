from __future__ import annotations
from typing import List, Set
import numpy as np
import pandas as pd
from ml.src.scoring import SkillExtractor
from ml.src.retrieval import bm25_full_groupwise, bm25_skills_groupwise, groupwise_minmax
from ml.src.embedder import SbertConfig, SbertEmbedder, add_sbert_similarity_feature, get_sbert_embedder


def build_features(
    df: pd.DataFrame,
    fuzzy_vendor: float = 0.70,
    max_alias_len: int = 4,
    laplace_a: float = 1.0,
    laplace_b: float = 1.0,
    bm25_full_weight: float = 0.5,
    bm25_skills_weight: float = 0.5,
    final_skill_weight: float = 0.5,
    final_bm25_weight: float  = 0.5,
    use_embedding: bool = True,
    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    emb_device: str = None,
    emb_batch_size: int = 64,
    emb_cache_path: str = None,
    emb_per_jd_norm: bool = True,
) -> pd.DataFrame:
    required = {"jd_id", "job_description_text", "cv_id", "resume_text"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Input df missing columns: {miss}")

    feats = df[["jd_id","job_description_text","cv_id","resume_text"]].copy()

    def _cov_smooth(hit_set: Set[str], req_set: Set[str]) -> float:
        n_hit = len(hit_set & req_set); n_req = len(req_set)
        return (n_hit + laplace_a) / (n_req + laplace_a + laplace_b) if n_req else 0.0

    extr = SkillExtractor(
        fuzzy_threshold=fuzzy_vendor,
        max_alias_len=max_alias_len,
        use_fuzzy=True,
    )

    n = len(feats)
    vendor_cov = np.zeros(n, dtype=float)
    cv_skills: List[Set[str]] = [set() for _ in range(n)]
    jd_skills: List[Set[str]] = [set() for _ in range(n)]

    for gid, g in feats.groupby("jd_id"):
        idx = g.index.values
        jd_text = str(g["job_description_text"].iloc[0])
        jd_all = set(extr.extract(jd_text))
        for row in idx:
            cv_text = str(feats.at[row, "resume_text"])
            cv_set  = set(extr.extract(cv_text))
            vendor_cov[row] = _cov_smooth(cv_set, jd_all)
            cv_skills[row]  = cv_set
            jd_skills[row]  = jd_all

    feats["vendor_cov"]  = vendor_cov
    feats["skill_score"] = vendor_cov 

    tmp = feats.copy()
    tmp["bm25_full"] = bm25_full_groupwise(
        tmp, resume_col="resume_text", jd_col="job_description_text", group_col="jd_id"
    )
    tmp["bm25_skills"] = bm25_skills_groupwise(
        tmp,
        cv_skills=pd.Series(cv_skills, index=tmp.index),
        jd_skills=pd.Series(jd_skills, index=tmp.index),
        group_col="jd_id",
    )
    feats["bm25_full_norm"]   = groupwise_minmax(tmp["bm25_full"].values,   feats["jd_id"].values)
    feats["bm25_skills_norm"] = groupwise_minmax(tmp["bm25_skills"].values, feats["jd_id"].values)

    w_bm = (bm25_full_weight + bm25_skills_weight) or 1.0
    wf, ws = bm25_full_weight / w_bm, bm25_skills_weight / w_bm
    feats["bm25_combo"] = wf * feats["bm25_full_norm"] + ws * feats["bm25_skills_norm"]

    w_fin = (final_skill_weight + final_bm25_weight) or 1.0
    w_s, w_b = final_skill_weight / w_fin, final_bm25_weight / w_fin
    feats["final_score"] = w_s * feats["skill_score"] + w_b * feats["bm25_combo"]

    if use_embedding:
        cfg = SbertConfig(
            model_name=emb_model_name,
            device=emb_device,
            batch_size=emb_batch_size,
            normalize=True,
            cache_path=emb_cache_path,
        )
        embedder = get_sbert_embedder(cfg)
        add_sbert_similarity_feature(
            feats,
            embedder=embedder,
            jd_col="job_description_text",
            cv_col="resume_text",
            out_col="emb_cosine",
            per_jd_norm=emb_per_jd_norm,
        )

    cols = ["jd_id","cv_id","vendor_cov","skill_score",
            "bm25_full_norm","bm25_skills_norm","bm25_combo","final_score"]
    if use_embedding:
        for c in ["emb_cosine","emb_cosine_norm"]:
            if c in feats.columns: cols.append(c)

    return feats[cols]