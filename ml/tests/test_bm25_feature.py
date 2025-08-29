import inspect
import numpy as np
import pandas as pd
import pytest
import ml.src.retrieval as bm25

@pytest.fixture
def df_bm25():
    jd = "Must-have: Python FastAPI PostgreSQL Docker."
    rows = [
        {"jd_id": "jd-1", "job_description_text": jd, "cv_id": "cv-1",
         "resume_text": "Python FastAPI PostgreSQL Docker; CI/CD GitHub Actions; AWS ECS"},
        {"jd_id": "jd-1", "job_description_text": jd, "cv_id": "cv-2",
         "resume_text": "Python Django MySQL; some Docker"},
        {"jd_id": "jd-1", "job_description_text": jd, "cv_id": "cv-3",
         "resume_text": "Java Spring; Oracle; no Python"},
    ]
    return pd.DataFrame(rows)

def _call_bm25_full_groupwise(df):
    """Gọi hàm với chữ ký linh hoạt (tránh lỗi 'unexpected keyword')."""
    f = bm25.bm25_full_groupwise
    sig = inspect.signature(f).parameters
    if {"text_col","query_col","group_col"} <= set(sig):
        return f(df, text_col="resume_text", query_col="job_description_text", group_col="jd_id")
    elif {"resume_col","jd_col","group_col"} <= set(sig):
        return f(df, resume_col="resume_text", jd_col="job_description_text", group_col="jd_id")
    elif {"df","resume_col","jd_col","group_col"} <= set(sig):
        return f(df=df, resume_col="resume_text", jd_col="job_description_text", group_col="jd_id")
    else:
        # fallback: positional by popular order
        return f(df, "resume_text", "job_description_text", "jd_id")

def _call_bm25_skills_groupwise(df, cv_skills, jd_skills):
    """Tương tự cho phiên bản skills."""
    f = bm25.bm25_skills_groupwise
    sig = inspect.signature(f).parameters
    if {"df","cv_skills","jd_skills","group_col"} <= set(sig):
        return f(df=df, cv_skills=cv_skills, jd_skills=jd_skills, group_col="jd_id")
    elif {"cv_skills","jd_skills","group_col"} <= set(sig):
        return f(df, cv_skills=cv_skills, jd_skills=jd_skills, group_col="jd_id")
    elif {"cv_skill_col","jd_skill_col","group_col"} <= set(sig):
        return f(df, cv_skill_col="cv_skills", jd_skill_col="jd_skills", group_col="jd_id")
    else:
        # fallback positional (df, cv_skills, jd_skills, group_col)
        return f(df, cv_skills, jd_skills, "jd_id")

def test_bm25_full_groupwise_ranking(df_bm25):
    scores = _call_bm25_full_groupwise(df_bm25)
    # expect: cv-1 (good) > cv-2 (average) > cv-3 (less relevant)
    assert len(scores) == len(df_bm25)
    order = np.argsort(-np.asarray(scores))  # descending
    ranked_cv = df_bm25.iloc[order]["cv_id"].tolist()
    assert ranked_cv[0] == "cv-1"
    assert ranked_cv[-1] == "cv-3"

def test_bm25_skills_groupwise(df_bm25):
    jd_sk = [["python","fastapi","postgresql","docker"]] * len(df_bm25)
    cv_sk = [
        ["python","fastapi","postgresql","docker"],
        ["python","docker"],
        ["java","spring"],
    ]
    cv_series = pd.Series(cv_sk, index=df_bm25.index, name="cv_skills")
    jd_series = pd.Series(jd_sk, index=df_bm25.index, name="jd_skills")

    scores = _call_bm25_skills_groupwise(df_bm25, cv_series, jd_series)
    assert len(scores) == len(df_bm25)

    # expect: cv-1 > cv-2 > cv-3
    order = np.argsort(-np.asarray(scores))
    ranked_cv = df_bm25.iloc[order]["cv_id"].tolist()
    assert ranked_cv[0] == "cv-1"
    assert ranked_cv[-1] == "cv-3"
