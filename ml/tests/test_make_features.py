import numpy as np
import pandas as pd
import pytest

import ml.src.features as mf

class _FakeSkillExtractor:
    def __init__(self, **kw): pass
    def extract(self, text: str):
        t = (text or "").lower()
        skills = []
        for k in ["python","fastapi","postgresql","docker","aws","java","kubernetes"]:
            if k in t: skills.append(k)
        return skills

def test_build_features_minimal(monkeypatch, tiny_pairs):
    monkeypatch.setattr(mf, "SkillExtractor", _FakeSkillExtractor)

    def fake_bm25_full_groupwise(df, resume_col, jd_col, group_col):
        vals = []
        for gid, g in df.groupby(group_col):
            n = len(g)
            sc = g[resume_col].str.count("python", case=False).astype(float)
            sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
            vals.append(sc.values)
        return np.concatenate(vals)

    def fake_bm25_skills_groupwise(df, cv_skills, jd_skills, group_col):
        vals = []
        for gid, g in df.groupby(group_col):
            out = []
            for idx in g.index:
                a = set(cv_skills.loc[idx])
                b = set(jd_skills.loc[idx])
                j = len(a & b) / (len(a | b) + 1e-12)
                out.append(j)
            vals.append(np.array(out, dtype=float))
        return np.concatenate(vals)

    monkeypatch.setattr(mf, "bm25_full_groupwise",  fake_bm25_full_groupwise)
    monkeypatch.setattr(mf, "bm25_skills_groupwise", fake_bm25_skills_groupwise)

    class _FakeSbert:
        def __init__(self, cfg): pass
        def encode(self, texts):
            out = []
            for t in texts:
                h = abs(hash((t or ""))) % 997
                v = np.array([h, h//7, h//13], dtype=float)
                v = v / (np.linalg.norm(v) + 1e-12)
                out.append(v)
            return np.vstack(out).astype(np.float32)


    try:
        import ml.src.embedder as emb
        def fake_add(df, embedder, jd_col, cv_col, out_col, per_jd_norm=True):
            jd_tbl = df[["jd_id", jd_col]].drop_duplicates("jd_id")
            cv_tbl = df[["cv_id", cv_col]].drop_duplicates("cv_id")
            E_jd = _FakeSbert(None).encode(jd_tbl[jd_col].astype(str).tolist())
            E_cv = _FakeSbert(None).encode(cv_tbl[cv_col].astype(str).tolist())
            jd2i = {j:i for i,j in enumerate(jd_tbl["jd_id"])}
            cv2i = {c:i for i,c in enumerate(cv_tbl["cv_id"])}
            sims = np.empty(len(df), dtype=np.float32)
            for i, r in enumerate(df.itertuples(index=False)):
                sims[i] = float((E_jd[jd2i[r.jd_id]] * E_cv[cv2i[r.cv_id]]).sum())
            df[out_col] = sims
            if per_jd_norm:
                df[out_col + "_norm"] = df.groupby("jd_id")[out_col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
                )
            return E_jd, E_cv
        monkeypatch.setattr(emb, "SbertEmbedder", _FakeSbert)
        monkeypatch.setattr(emb, "add_sbert_similarity_feature", fake_add)
    except ImportError:
        print('Fail to load embedder')
        pass

    feats = mf.build_features(
        df=tiny_pairs.copy(),
        fuzzy_vendor=0.7,
        max_alias_len=4,
        use_embedding=True,
        emb_per_jd_norm=True,
    )

    expect_cols = {"jd_id","cv_id","vendor_cov","bm25_full_norm","bm25_skills_norm","bm25_combo","final_score"}
    assert expect_cols.issubset(set(feats.columns))

    if "emb_cosine_norm" in feats.columns:
        assert feats["emb_cosine_norm"].between(0.0, 1.0).all()

    ranked = feats.sort_values("final_score", ascending=False)
    assert ranked.iloc[0]["cv_id"] in {"cv-1","cv-3","cv-5"}
