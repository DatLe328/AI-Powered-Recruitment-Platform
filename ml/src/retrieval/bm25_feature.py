from __future__ import annotations
from typing import List, Dict, Iterable, Optional, Sequence, Tuple
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

EN_STOP = {
    # function words
    "the","a","an","and","or","in","on","for","of","to","with","at","as","by","is","are","was","were",
    "be","been","being","this","that","these","those","from","into","within","via","using","use","it",
    "we","you","they","our","their","your","i","he","she","his","her","them","me","my","us","also",
    # cv/jd common noise
    "responsible","responsibilities","experienced","experience","experiences","familiar","knowledge",
    "skills","skill","ability","abilities","proficient","expert","strong","good","excellent","great",
    "working","work","worked","team","teams","environment","environments","company","companies",
    "project","projects","role","roles","tasks","task","duties","duty","objective","objectives",
    "summary","description","descriptions","requirement","requirements","preferred","plus","nice",
    "including","etc","etc.","eg","e.g","ie","i.e","based","per","performs","perform","performing",
    "developing","develop","developed","build","building","built","design","designing","designed",
    "implement","implementation","implemented","maintain","maintaining","maintained","support",
    "supporting","supported","lead","leading","led","manage","managing","managed","mentor","mentoring",
    "coordinate","coordinating","coordinated","collaborate","collaboration","collaborated",
    "degree","bachelor","master","phd","university","college","certification","certificate",
    "junior","senior","intern","internship","full-time","part-time","contract","freelance","remote",
    # time/date
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","march","april","may","june","july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec","year","years","month","months",
}

CANON_REPLACEMENTS = {
    r"\bk8s\b": "kubernetes",
    r"\bgolang\b": "go",
    r"\bci/?cd\b": "ci cd",
    r"\bnode\.?js\b": "node.js",
    r"\breact\.?js\b": "react.js",
    r"\bnext\.?js\b": "next.js",
    r"\b\.net\b": ".net",
    r"\bc#\b": "c#",
    r"\bc\+\+\b": "c++",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._/#-]*", re.IGNORECASE)
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
TAG_RE   = re.compile(r"<[^>]+>")

def _clean_text(s: str) -> str:
    s = str(s or "")
    # html/url/email removal
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    # whitespace & lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    # light canonicalization
    for pat, rep in CANON_REPLACEMENTS.items():
        s = re.sub(pat, rep, s)
    return s

def tokenize(text: str, stop: Optional[set] = None) -> List[str]:
    stop = EN_STOP if stop is None else stop
    s = _clean_text(text)
    toks = TOKEN_RE.findall(s)
    return [t for t in toks if t not in stop]

# ---------------------------------------------------------------------
# Core BM25 builders
# ---------------------------------------------------------------------

def _bm25_scores_for_group(
    corpus_tokens: List[List[str]],
    query_tokens: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    if not corpus_tokens:
        return np.zeros(0, dtype=float)
    if not query_tokens:
        return np.zeros(len(corpus_tokens), dtype=float)
    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    return bm25.get_scores(query_tokens).astype(float)

def bm25_full_groupwise(
    df: pd.DataFrame,
    resume_col: str = "resume_text",
    jd_col: str = "job_description_text",
    group_col: str = "jd_id",
    *,
    analyzer=tokenize,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """
    BM25 (Okapi) per JD group.
    - corpus: tokenized resumes within each group
    - query : tokenized JD (first row in group)
    Returns: np.ndarray aligned to df.index
    """
    scores = np.zeros(len(df), dtype=float)
    if len(df) == 0:
        return scores
    for gid, g in df.groupby(group_col):
        corpus = [analyzer(x) for x in g[resume_col].tolist()]
        query  = analyzer(str(g[jd_col].iloc[0]))
        s = _bm25_scores_for_group(corpus, query, k1=k1, b=b)
        scores[g.index.values] = s if len(s) else 0.0
    return scores

def bm25_skills_groupwise(
    df: pd.DataFrame,
    cv_skills: pd.Series,
    jd_skills: pd.Series,
    group_col: str = "jd_id",
    *,
    k1: float = 1.5,
    b: float = 0.75,
    sort_skills: bool = True,
) -> np.ndarray:
    """
    BM25 on skills only.
    - corpus: per-CV list of skills (canonical strings)
    - query : set/list of JD skills for the group
    Notes:
      * if values are sets, we convert to sorted lists for stable scoring.
    """
    scores = np.zeros(len(df), dtype=float)
    if len(df) == 0:
        return scores
    for gid, g in df.groupby(group_col):
        idx = g.index
        # normalize corpus skills
        corpus = []
        for x in cv_skills.loc[idx]:
            if isinstance(x, (set, frozenset)):
                lst = sorted(x) if sort_skills else list(x)
            elif isinstance(x, list):
                lst = x
            else:
                lst = []
            corpus.append(lst)
        # normalize query skills
        qraw = jd_skills.loc[idx].iloc[0] if len(idx) else []
        if isinstance(qraw, (set, frozenset)):
            query = sorted(qraw) if sort_skills else list(qraw)
        elif isinstance(qraw, list):
            query = qraw
        else:
            query = []
        s = _bm25_scores_for_group(corpus, query, k1=k1, b=b)
        scores[idx] = s if len(s) else 0.0
    return scores

def bm25_weighted_groupwise(
    df: pd.DataFrame,
    resume_col: str = "resume_text",
    jd_col: str = "job_description_text",
    group_col: str = "jd_id",
    term_weights_map: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    analyzer=tokenize,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """
    Weighted BM25: up-weight important JD terms by repeating them in the query.
    term_weights_map: mapping { jd_id -> {term -> weight(float)} }
      if None -> falls back to bm25_full_groupwise
    """
    if term_weights_map is None:
        return bm25_full_groupwise(df, resume_col, jd_col, group_col, analyzer=analyzer, k1=k1, b=b)

    def apply_weights(tokens: List[str], weights: Dict[str, float]) -> List[str]:
        out: List[str] = []
        for t in tokens:
            w = int(round(float(weights.get(t, 1.0))))
            out.extend([t] * max(1, w))
        return out

    scores = np.zeros(len(df), dtype=float)
    if len(df) == 0:
        return scores
    for gid, g in df.groupby(group_col):
        corpus = [analyzer(x) for x in g[resume_col].tolist()]
        base_query = analyzer(str(g[jd_col].iloc[0]))
        tw = term_weights_map.get(str(gid)) or term_weights_map.get(gid) or {}
        query = apply_weights(base_query, tw) if tw else base_query
        s = _bm25_scores_for_group(corpus, query, k1=k1, b=b)
        scores[g.index.values] = s if len(s) else 0.0
    return scores

# ---------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------

def groupwise_minmax(scores: np.ndarray, groups: Iterable) -> np.ndarray:
    """Min–max per group → [0..1]."""
    scores = np.asarray(scores, dtype=float).copy()
    groups = np.asarray(list(groups))
    if scores.size == 0:
        return scores
    for gid in np.unique(groups):
        idx = (groups == gid)
        if idx.sum() <= 1:
            scores[idx] = 0.0
            continue
        s = scores[idx]
        lo, hi = float(np.min(s)), float(np.max(s))
        scores[idx] = (s - lo) / (hi - lo + 1e-12) if hi > lo else 0.0
    return scores

def groupwise_norm(scores: np.ndarray, groups: Iterable, method: str = "minmax") -> np.ndarray:
    """
    General group-wise normalization.
    method ∈ {"minmax", "zscore", "softmax", "rank", "none"}
    """
    scores = np.asarray(scores, dtype=float).copy()
    groups = np.asarray(list(groups))
    if method == "none" or scores.size == 0:
        return scores

    out = np.zeros_like(scores)
    for gid in np.unique(groups):
        idx = (groups == gid)
        s = scores[idx]
        if s.size <= 1:
            out[idx] = 0.0
            continue
        if method == "minmax":
            lo, hi = float(np.min(s)), float(np.max(s))
            out[idx] = (s - lo) / (hi - lo + 1e-12) if hi > lo else 0.0
        elif method == "zscore":
            mu, sd = float(np.mean(s)), float(np.std(s) + 1e-12)
            out[idx] = (s - mu) / sd
        elif method == "softmax":
            x = s - float(np.max(s))
            ex = np.exp(x)
            out[idx] = ex / (ex.sum() + 1e-12)
        elif method == "rank":
            # normalized rank in [0,1]
            order = s.argsort(kind="mergesort")  # stable
            r = np.empty_like(order, dtype=float)
            r[order] = np.linspace(0.0, 1.0, num=s.size, endpoint=True)
            out[idx] = r
        else:
            raise ValueError(f"Unknown method={method}")
    return out

# ---------------------------------------------------------------------
# Combo helper
# ---------------------------------------------------------------------

def bm25_dual_groupwise(
    df: pd.DataFrame,
    *,
    resume_col: str = "resume_text",
    jd_col: str = "job_description_text",
    group_col: str = "jd_id",
    cv_skills: Optional[pd.Series] = None,
    jd_skills: Optional[pd.Series] = None,
    w_full: float = 0.5,
    w_skills: float = 0.5,
    norm: str = "minmax",
    analyzer=tokenize,
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: compute both full-text BM25 and skills-BM25, normalize per group,
    and return the weighted combo.
    """
    s_full = bm25_full_groupwise(df, resume_col, jd_col, group_col, analyzer=analyzer, k1=k1, b=b)
    if cv_skills is None or jd_skills is None:
        s_sk = np.zeros_like(s_full)
    else:
        s_sk = bm25_skills_groupwise(df, cv_skills, jd_skills, group_col, k1=k1, b=b)

    sn_full = groupwise_norm(s_full, df[group_col].values, method=norm)
    sn_sk   = groupwise_norm(s_sk,   df[group_col].values, method=norm)

    w = (w_full + w_skills) or 1.0
    mix = (w_full / w) * sn_full + (w_skills / w) * sn_sk
    return sn_full, sn_sk, mix

# ---------------------------------------------------------------------
# (Optional) simple self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    jd = "We seek a Cloud DevOps Engineer: Docker, Kubernetes, AWS, Git. Nice: Python, Terraform."
    cvs = [
        "DevOps with Docker, Kubernetes, AWS, Git, Terraform, Python.",
        "Backend Java/Python, Git, some AWS. No Docker/Kubernetes yet.",
        "SRE with Kubernetes, Docker, GitOps; strong AWS and Terraform.",
    ]
    df = pd.DataFrame({
        "jd_id": ["demo"]*3,
        "job_description_text": [jd]*3,
        "resume_text": cvs
    })
    # full BM25
    s_full = bm25_full_groupwise(df)
    # fake skills just for demo
    cv_sk = pd.Series([{"docker","kubernetes","aws","git","terraform","python"},
                       {"java","python","git","aws"},
                       {"kubernetes","docker","aws","terraform"}])
    jd_sk = pd.Series([{"docker","kubernetes","aws","git","python","terraform"}]*3)
    s_sk = bm25_skills_groupwise(df, cv_sk, jd_sk)
    s_full_n, s_sk_n, s_mix = bm25_dual_groupwise(df, cv_skills=cv_sk, jd_skills=jd_sk, w_full=0.4, w_skills=0.6)
    print("Scores:", s_full, s_sk, s_mix)


def test():
    from scoring import SkillExtractor
    extr = SkillExtractor(fuzzy_threshold=0.70)

    jd_text = """
    We are looking for a Backend Engineer with strong Python, FastAPI, Docker, PostgreSQL,. Knowledge of AWS is a plus. 
    """

    cv_list = [
        "Backend engineer with 4+ years in Python, FastAPI, Docker, PostgreSQL, GitHub Actions CI/CD.",
        "Frontend developer React/Next.js, UI/UX focus. Basic Node.js.",
        "Java Spring Boot microservices; some Python scripting and Kafka."
    ]

    df = pd.DataFrame({
        "jd_id": ["JD_DEMO"] * len(cv_list),
        "job_description_text": [jd_text] * len(cv_list),
        "resume_text": cv_list
    })

    # --- 4. Extract skills ---
    df["cv_skills"] = df["resume_text"].apply(lambda x: extr.extract(x))
    jd_skills = extr.extract(jd_text)
    df["jd_skills"] = [jd_skills] * len(df)

    # --- 5. BM25 features ---
    df["bm25_full"] = bm25_full_groupwise(df, "resume_text", "job_description_text", "jd_id")
    df["bm25_skills"] = bm25_skills_groupwise(df, df["cv_skills"], df["jd_skills"], "jd_id")

    df["bm25_full_norm"] = groupwise_minmax(df["bm25_full"].values, df["jd_id"].values)
    df["bm25_skills_norm"] = groupwise_minmax(df["bm25_skills"].values, df["jd_id"].values)
    df["bm25_combo"] = 0.3*df["bm25_full_norm"] + 0.7*df["bm25_skills_norm"]

    # --- 6. Show ranking ---
    print("\n=== Ranking by bm25_combo ===")
    print(df.sort_values("bm25_combo", ascending=False)[
        ["resume_text", "cv_skills", "bm25_full_norm", "bm25_skills_norm", "bm25_combo"]
    ])

if __name__ == "__main__":
    test()