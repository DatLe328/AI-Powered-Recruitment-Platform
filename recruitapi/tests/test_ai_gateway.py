import types
import pandas as pd
import pytest

# import module gateway
from ml import apis

class FakeRanker:
    """Giả lập XGBRanker: rank theo số lần 'python' trừ số lần 'java'."""
    def rank(self, df_pairs: pd.DataFrame, topk=None) -> pd.DataFrame:
        s = df_pairs["resume_text"].str.count(r"\bpython\b", case=False) \
            - df_pairs["resume_text"].str.count(r"\bjava\b", case=False)
        out = df_pairs[["jd_id","cv_id"]].copy()
        out["pred"] = s.astype(float)
        out = out.sort_values("pred", ascending=False).reset_index(drop=True)
        if topk:
            out = out.head(int(topk))
        return out

def test_rank_cv_for_jd_monkeypatched(monkeypatch):
    # reset model cache
    monkeypatch.setattr(apis, "_MODEL", None)

    # patch: CVJDXGBRanker.load -> FakeRanker()
    monkeypatch.setattr(apis.CVJDXGBRanker, "load", staticmethod(lambda m, c: FakeRanker()))

    # call gateway
    jd_req = "Must-have: Python and SQL"
    jd_desc = "We prefer experience with APIs"
    cands = [
        {"cv_id":"cv-1","resume_text":"Python, FastAPI, SQL"},
        {"cv_id":"cv-2","resume_text":"Java, Spring"},
        {"cv_id":"cv-3","resume_text":"Python, Java, SQL"},
        {"cv_id":"cv-4","resume_text":"C++ only"},
        {"cv_id":"cv-5","resume_text":"python python (two hits)"},
    ]
    results = apis.rank_cv_for_jd(
        job_requirement=jd_req,
        job_description=jd_desc,
        candidates=cands,
        topk=None,
    )

    # kiểm tra cấu trúc & thứ tự
    assert isinstance(results, list) and len(results) == 5
    # theo rule: cv-5 (2 python) > cv-1 (1) > cv-3 (1-1=0) > cv-4 (0) > cv-2 (-1)
    order = [r["cv_id"] for r in results]
    assert order == ["cv-5","cv-1","cv-3","cv-4","cv-2"]
    # có đủ trường
    assert {"jd_id","cv_id","pred","score"} <= set(results[0].keys())
    # score == pred (gateway không bonus)
    assert all(abs(r["score"] - r["pred"]) < 1e-9 for r in results)

def test_topk(monkeypatch):
    monkeypatch.setattr(apis, "_MODEL", None)
    monkeypatch.setattr(apis.CVJDXGBRanker, "load", staticmethod(lambda m, c: FakeRanker()))

    results = apis.rank_cv_for_jd(
        job_requirement="Python",
        job_description="",
        candidates=[{"resume_text":"python"}, {"resume_text":"java"}, {"resume_text":"python python"}],
        topk=2,
    )
    assert len(results) == 2
    assert [r["cv_id"] for r in results] == [results[0]["cv_id"], results[1]["cv_id"]]  # có 2 phần tử
