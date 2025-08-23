from functools import lru_cache
import json
import os
from pathlib import Path
import json, re


def merge_skill_dicts(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, [])
            out[k].extend(v)
    # de-dup giữ thứ tự (case-insensitive)
    for k, v in list(out.items()):
        seen=set(); clean=[]
        for s in v:
            s2=" ".join((s or "").split()).strip()
            if s2 and s2.lower() not in seen:
                seen.add(s2.lower()); clean.append(s2)
        out[k]=clean
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
    """
    Input schema (per file):
      { category: { canonical: [aliases...] , ... }, ... }
    Return flat dict: canonical -> [aliases...]
    """
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
    # giữ vài trường hợp đặc biệt ngắn
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
    """
    Load skill dict from a single JSON file OR merge all *.json in a folder.
    Returns a flat dict: canonical -> [aliases...]
    """
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


"""
    ESCO
"""

def build_skill_dict_from_etl(labels_path, min_alias_len=2):
    labels_by_skill = json.load(open(labels_path, encoding="utf-8"))  # {skill_id: [labels...]}

    def clean(lst):
        out = []
        for s in lst:
            parts = re.split(r"(?:\r?\n)|(?:\\n)", s or "")
            for p in parts:
                p = re.sub(r"\s+", " ", p).strip()
                if len(p) >= min_alias_len:
                    out.append(p)
        # de-dup giữ thứ tự (case-insensitive)
        seen=set(); clean=[]
        for x in out:
            xl=x.lower()
            if xl not in seen:
                seen.add(xl); clean.append(x)
        return clean

    skill_dict = {}
    for sid, labels in labels_by_skill.items():
        labels = clean(labels)
        if not labels:
            continue
        canon = labels[0]               # preferred label làm key
        if canon not in skill_dict:
            skill_dict[canon] = labels  # preferred + alt
        else:
            # gộp nếu trùng preferred giữa 2 skill_id (hiếm)
            skill_dict[canon] = clean(skill_dict[canon] + labels)
    return skill_dict

def slugify(s: str) -> str:
    try:
        from unidecode import unidecode
        s = unidecode(s or "")
    except Exception:
        s = s or ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def tokenize(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]

import csv
def load_occupations_corpus(tsv_path: str):
    rows = []
    # utf-8-sig để xử lý BOM trên Windows; newline='' để csv xử lý CRLF chuẩn
    with open(tsv_path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        # sanity check cột
        need = {"occ_id", "text"}
        have = set(rdr.fieldnames or [])
        if not need.issubset(have):
            raise ValueError(
                f"File '{tsv_path}' thiếu cột {need}. Hiện có: {sorted(have)}.\n"
                "Bạn có đang trỏ nhầm sang skills_corpus.tsv không?"
            )

        for i, row in enumerate(rdr, start=2):  # start=2 vì dòng 1 là header
            occ_id = (row.get("occ_id") or "").strip()
            text   = (row.get("text") or "").strip()
            if not occ_id or not text:
                # bỏ qua dòng rỗng / hỏng
                continue
            # label = phần trước ' | '
            label = text.split(" | ", 1)[0].strip()
            # tạo slug
            try:
                from unidecode import unidecode
                s = unidecode(label)
            except Exception:
                s = label
            slug = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "unknown"

            rows.append({"occ_id": occ_id, "label": label, "text": text, "slug": slug})

    if not rows:
        raise ValueError(f"Không đọc được dòng hợp lệ nào từ '{tsv_path}'. Kiểm tra file có đúng là occupations_corpus.tsv không.")
    return rows

def score_occ_simple(jd_text: str, occ_rows, topk=5):
    # token-overlap + char-sim (SequenceMatcher) đơn giản
    import difflib
    q = " ".join(tokenize(jd_text))
    q_tokens = set(tokenize(jd_text))

    scored = []
    for r in occ_rows:
        t = r["label"] + " " + (r["text"] or "")
        toks = set(tokenize(t))
        # Jaccard nhẹ
        inter = len(q_tokens & toks)
        union = len(q_tokens | toks) or 1
        jacc = inter / union
        # char similarity
        char_sim = difflib.SequenceMatcher(None, r["label"].lower(), q.lower()).ratio()
        score = 0.6 * char_sim + 0.4 * jacc
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

def score_occ_tfidf(jd_text: str, occ_rows, topk=5):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
    except Exception:
        return score_occ_simple(jd_text, occ_rows, topk=topk)

    docs = [r["label"] + " " + (r["text"] or "") for r in occ_rows]
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=1, max_df=0.8)
    X = vec.fit_transform(docs)
    qv = vec.transform([jd_text])
    sims = linear_kernel(qv, X).ravel()
    idxs = sims.argsort()[::-1][:topk]
    return [(float(sims[i]), occ_rows[i]) for i in idxs]


# ---------- load pack theo slug ----------
@lru_cache(maxsize=256)
def load_pack_subset(big_pack_path: str, slug: str):
    # big pack: {slug: {...}, slug2: {...}, ...}
    data = json.load(open(big_pack_path, encoding="utf-8"))
    if slug not in data:
        raise KeyError(f"Slug '{slug}' không có trong pack.")
    return {slug: data[slug]}

def merge_skill_dicts(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, [])
            out[k].extend(v)
    # de-dup giữ thứ tự (case-insensitive)
    for k, v in list(out.items()):
        seen=set(); clean=[]
        for s in v:
            s2=" ".join((s or "").split()).strip()
            if s2 and s2.lower() not in seen:
                seen.add(s2.lower()); clean.append(s2)
        out[k]=clean
    return out

def pack_to_skill_dict(pack_body, include="both"):
    buckets = {"both": ("essential","optional"), "essential": ("essential",), "optional": ("optional",)}
    d = {}
    for b in buckets[include]:
        for canon, aliases in pack_body.get(b, {}).items():
            d.setdefault(canon, []).extend(aliases)
    # clean
    return merge_skill_dicts(d)

# ---------- pipeline chọn top-K slug rồi load ----------
def build_skill_dict_for_jd(jd_text: str,
                            occ_tsv_path: str,
                            big_pack_path: str,
                            topk=3,
                            include="both",
                            scorer="tfidf"):
    occ_rows = load_occupations_corpus(occ_tsv_path)
    scored = score_occ_tfidf(jd_text, occ_rows, topk=topk) if scorer=="tfidf" else score_occ_simple(jd_text, occ_rows, topk=topk)
    slugs = [r["slug"] for _, r in scored]

    # gộp các pack theo slugs
    skill_dict = {}
    for slug in slugs:
        subset = load_pack_subset(big_pack_path, slug)
        body = subset[slug]
        skill_dict = merge_skill_dicts(skill_dict, pack_to_skill_dict(body, include=include))
    print("Splug list:", slugs)
    return skill_dict, slugs


def load_esco_buckets(big_pack_path, slugs):
    """Trả về 2 set canonical: essential_set, optional_set (union của slugs)."""
    data = json.load(open(big_pack_path, encoding="utf-8"))
    essential, optional = set(), set()
    for slug in slugs:
        body = data.get(slug, {})
        essential |= set(body.get("essential", {}).keys())
        optional  |= set(body.get("optional", {}).keys())
    # nếu 1 skill nằm ở cả 2, ưu tiên essential
    optional -= essential
    return essential, optional