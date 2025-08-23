from __future__ import annotations
import html
import json
import re
import time
import unicodedata
import itertools
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable, Optional
import os
from pathlib import Path
import numpy as np


try:
    from sentence_transformers import SentenceTransformer 
    _HAS_ST = True
except Exception:
    _HAS_ST = False


EN_STOP = {
    # base function words
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

# helpful canonical replacements before matching
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

def clean_text(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    for pat, rep in CANON_REPLACEMENTS.items():
        s = re.sub(pat, rep, s)
    return s

def tokens(text: str) -> List[str]:
    text = clean_text(text)
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in EN_STOP]

def ngrams(seq: List[str], n: int) -> List[str]:
    return [" ".join(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())

def _unique_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _flatten_one(obj):
    flat = {}
    if not isinstance(obj, dict):
        return flat
    for _, mapping in obj.items():
        if not isinstance(mapping, dict):
            continue
        for canon, aliases in mapping.items():
            if isinstance(aliases, (list, tuple, set)):
                a = [str(x) for x in aliases if isinstance(x, str)]
            elif isinstance(aliases, str):
                a = [aliases]
            else:
                a = []
            canon_n = _norm(canon)
            alias_n = [_norm(x) for x in ([canon] + a) if isinstance(x, str) and x.strip()]
            alias_n = _unique_keep_order([x for x in alias_n if x])
            if not canon_n or not alias_n:
                continue
            if canon_n in flat:
                flat[canon_n] = _unique_keep_order(flat[canon_n] + alias_n)
            else:
                flat[canon_n] = alias_n
    return flat

def _is_ascii_english(s: str) -> bool:
    try:
        s.encode("ascii")
    except Exception:
        return False
    low = s.strip().lower()
    if len(low) <= 2 and low not in {"c#", "c++", "go"}:
        return False
    return True

def _filter_english_only(d):
    out = {}
    for canon, aliases in d.items():
        als = [a for a in aliases if _is_ascii_english(a)]
        canon_ok = _is_ascii_english(canon)
        if not canon_ok and not als:
            continue
        canon_final = canon if canon_ok else als[0]
        out[canon_final] = _unique_keep_order([canon_final] + als)
    return out

def get_skill_dict(path_or_dir: str, english_only: bool = False):
    p = Path(path_or_dir)
    files = []
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No *.json found in folder: {p}")
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"Path not found: {p}")

    merged = {}
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            try:
                obj = json.load(fh)
            except Exception as e:
                print(f"[WARN] Skip {f.name}: {e}")
                continue
        flat = _flatten_one(obj)
        # merge
        for k, v in flat.items():
            if k in merged:
                merged[k] = _unique_keep_order(merged[k] + v)
            else:
                merged[k] = list(v)

    if english_only:
        merged = _filter_english_only(merged)
    return merged


_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}
_CANON_VEC_CACHE: Dict[Tuple[str, Tuple[str, ...]], np.ndarray] = {}

def _load_model(model_name: str) -> Optional["SentenceTransformer"]:
    if not _HAS_ST:
        return None
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


@dataclass
class ExtractorConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fuzzy_threshold: float = 0.70
    max_alias_len: int = 4
    use_fuzzy: bool = True
    max_fuzzy_candidates: int = 500

class SkillExtractor:
    def __init__(self,**kwargs):
        init_time = time.time()
        self.cfg = ExtractorConfig(**kwargs)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'vendor')
        skill_dict = get_skill_dict(data_path)
        self.skill_dict = self._normalize_skill_dict(skill_dict)
        self.alias2canon: Dict[str, str] = {}
        self._alias_re = self._build_alias_regex()
        self.canonical_names = sorted(self.skill_dict.keys())

        # Fuzzy backend
        self.model = _load_model(self.cfg.model_name) if self.cfg.use_fuzzy else None
        self.canon_vecs = self._embed_canon_skills() if (self.cfg.use_fuzzy and self.model is not None) else None
        print(f"[SkillExtractor] Model loading took: {time.time() - init_time:.4f} seconds")


    def _normalize_skill_dict(self, sd: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for k, v in sd.items():
            canon = clean_text(k)
            aliases = [clean_text(x) for x in ([k] + (v or []))]
            out[canon] = unique_keep_order(aliases)
        return out

    def _build_alias_regex(self) -> Optional[re.Pattern]:
        pats: List[str] = []
        for canon, aliases in self.skill_dict.items():
            for a in aliases:
                a = a.strip()
                if not a:
                    continue
                self.alias2canon[a] = canon
                pats.append(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])")
        return re.compile("|".join(pats), re.IGNORECASE) if pats else None

    def _embed_canon_skills(self) -> np.ndarray:
        key = (self.cfg.model_name, tuple(self.canonical_names))
        if key in _CANON_VEC_CACHE:
            return _CANON_VEC_CACHE[key]
        if self.model is None:
            return np.empty((0, 0), dtype=np.float32)

        alias_lists = [self.skill_dict[c] for c in self.canonical_names]
        flat = list(itertools.chain.from_iterable(alias_lists))
        emb = self.model.encode(flat, convert_to_numpy=True, normalize_embeddings=True)
        out = []
        i = 0
        for aliases in alias_lists:
            k = len(aliases)
            out.append(emb[i:i+k].mean(axis=0))
            i += k
        mat = np.vstack(out)
        _CANON_VEC_CACHE[key] = mat
        return mat

    def _gen_candidates(self, text: str) -> List[str]:
        toks = tokens(text)
        grams: List[str] = []
        maxn = max(1, min(self.cfg.max_alias_len, 4))
        for n in (1, 2, 3, maxn):
            grams.extend(ngrams(toks, n))
        grams = unique_keep_order(grams)
        good = []
        for g in grams:
            ws = g.split()
            if not ws:
                continue
            if all(w in EN_STOP for w in ws):
                continue
            if len(g) < 2 or len(g) > 64:
                continue
            good.append(g)
        good = [c for c in good if c not in self.alias2canon]
        if len(good) > self.cfg.max_fuzzy_candidates:
            good = good[: self.cfg.max_fuzzy_candidates]
        return good

    def extract(self, text: str) -> Set[str]:
        t0 = time.time()
        txt = clean_text(text)
        out: Set[str] = set()

        if self._alias_re:
            for m in self._alias_re.finditer(txt):
                alias = m.group(0).lower()
                canon = self.alias2canon.get(alias)
                if canon:
                    out.add(canon)

        if self.cfg.use_fuzzy and self.model is not None and self.canon_vecs is not None and self.canon_vecs.size:
            cands = self._gen_candidates(txt)
            if cands:
                cand_vecs = self.model.encode(cands, convert_to_numpy=True, normalize_embeddings=True)
                sims = np.einsum("ij,kj->ik", cand_vecs, self.canon_vecs)  # cosine (vectors are unit-normalized)
                best_idx = sims.argmax(axis=1)
                best_sim = sims.max(axis=1)
                thr = float(self.cfg.fuzzy_threshold)
                for cand, j, s in zip(cands, best_idx, best_sim):
                    if s >= thr:
                        out.add(self.canonical_names[j])
        print(f"[SkillExtractor] found={len(out)} in {time.time()-t0:.3f}s")
        return out

    def overlap(self, cv_text: str, jd_text: str) -> Tuple[Set[str], Set[str], Set[str]]:
        cv = self.extract(cv_text)
        jd = self.extract(jd_text)
        return cv, jd, (cv & jd)

def test_sep():

    jd = """
Experienced in PYTHON3 and Fast API; using Postgres + Docker on Amazon Web Services.
    """
    cv = """
SRE with Kubernetes, Docker, GitOps; strong AWS and Terraform. Automated deployments and used IaC with Terraform.
    """

    extr = SkillExtractor(fuzzy_threshold=0.70, max_alias_len=4)
    cv_skills, jd_skills, common = extr.overlap(cv, jd)
    print("CV skills     :", sorted(cv_skills))
    print("JD skills     :", sorted(jd_skills))
    print("Overlap (match):", sorted(common))

if __name__ == "__main__":
    test_sep()