import re
import pandas as pd
from ml import apis


"""Rank by number of occurrences of 'python' (case insensitive) minus 'java'."""
class FakeRanker:
    def rank(self, df_pairs: pd.DataFrame, topk=None) -> pd.DataFrame:
        s = (
            df_pairs["resume_text"].str.count(r"\bpython\b", flags=re.IGNORECASE)
            - df_pairs["resume_text"].str.count(r"\bjava\b",   flags=re.IGNORECASE)
        )
        out = df_pairs[["jd_id","cv_id"]].copy()
        out["pred"] = s.astype(float)
        out = out.sort_values("pred", ascending=False).reset_index(drop=True)
        if topk:
            out = out.head(int(topk))
        return out

def test_rank_cv_for_jd(monkeypatch):
    monkeypatch.setattr(apis, "_MODEL", None)
    monkeypatch.setattr(apis.CVJDXGBRanker, "load", staticmethod(lambda *a, **kw: FakeRanker()))

    results = apis.rank_cv_for_jd(
        job_requirement="Must-have: Python, FastAPI, PostgreSQL",
        job_description="We build APIs and CI/CD.",
        candidates=[
            {"cv_id":"cv-1","resume_text":"Python FastAPI Postgres"},
            {"cv_id":"cv-2","resume_text":"Java Spring"},
            {"cv_id":"cv-3","resume_text":"some python and java"},
        ],
        topk=None,
    )

    assert isinstance(results, list) and len(results) == 3
    assert {"jd_id","cv_id","pred","score"} <= set(results[0].keys())

    # expected order: cv-1 (1) > cv-3 (1-1=0) > cv-2 (-1)
    assert [r["cv_id"] for r in results] == ["cv-1","cv-3","cv-2"]
    # score == pred
    assert all(abs(r["score"] - r["pred"]) < 1e-9 for r in results)

def test_topk(monkeypatch):
    monkeypatch.setattr(apis, "_MODEL", None)
    monkeypatch.setattr(apis.CVJDXGBRanker, "load", staticmethod(lambda *a, **kw: FakeRanker()))

    results = apis.rank_cv_for_jd(
        job_requirement="Python",
        job_description="",
        candidates=[{"resume_text":"python"}, {"resume_text":"java"}, {"resume_text":"python python"}],
        topk=2,
    )
    assert len(results) == 2
    assert results[0]["pred"] >= results[1]["pred"]
