from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd

# ---------------------------- Logging ---------------------------------
def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------- Cleaning ---------------------------------
PARENS_BLOCK_RE = re.compile(r"\s*[\(\[\{][^)\]\}}]*[\)\]\}]")
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")
UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{2,}\b")

def clean_label(raw: str) -> str:
    s = (raw or "").lower()
    prev = None
    while prev != s:
        prev = s
        s = PARENS_BLOCK_RE.sub("", s)
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

# ---------------------------- NLP Fallbacks ----------------------------
def try_import_nltk():
    try:
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        return PorterStemmer(), WordNetLemmatizer()
    except Exception:
        return None, None


def stem_text(text: str, porter) -> str:
    tokens = text.split()
    return " ".join(porter.stem(t) for t in tokens)

def lemmatize_text(text: str, wn) -> str:
    tokens = text.split()
    out = [wn.lemmatize(t, pos="v") for t in tokens]
    out = [wn.lemmatize(t, pos="n") for t in out]
    out = [wn.lemmatize(t, pos="a") for t in out]
    return " ".join(out)

# ---------------------------- Abbreviation ---------------------
"""
    - Auto get abbr from name
    - Example: 
        + Purified Protein Derivative (PPD) -> PPD
        + Registered Physician In Vascular Interpretation (RPVI) -> RPVI
    - Error:
        + PRINCE2 (PRojects IN Controlled Environments 2) -> IN
"""
def infer_abbreviation_from_name(raw_name: str) -> str:
    if not raw_name:
        return ""
    m = re.search(r"\(([^)]+)\)", raw_name)
    if m:
        abbr = m.group(1).strip()
        # Chỉ lấy nếu abbr là một từ, toàn chữ hoa hoặc số, không chứa khoảng trắng
        if re.fullmatch(r"[A-Z0-9\-]+", abbr):
            return abbr
    # Nếu không có abbr hợp lệ, không lấy token viết hoa trong ngoặc nữa
    return ""

# ---------------------------- Config & Main ----------------------------
@dataclass
class Columns:
    id: str = "id"
    name: str = "name"
    type_name: Optional[str] = "type.name"


def validate_columns(df: pd.DataFrame, cols: Columns) -> None:
    missing = [c for c in [cols.id, cols.name] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s): {', '.join(missing)}")

def compute_unique_tokens(cleaned_series: pd.Series) -> Dict[str, int]:
    from collections import Counter
    cnt = Counter(tok for s in cleaned_series for tok in s.split())
    return dict(cnt)

def build_processed(
    df: pd.DataFrame, cols: Columns,
    include_unique_token: bool = True,
    infer_abbr: bool = True
) -> Dict[str, Dict[str, Any]]:
    df = df.copy()
    df["skill_cleaned"] = df[cols.name].astype(str).map(clean_label)
    df["skill_len"] = df["skill_cleaned"].str.split().str.len()

    porter, wn = try_import_nltk()
    logging.debug("NLTK Porter: %s; WordNetLemmatizer: %s", bool(porter), bool(wn))

    df["skill_stemmed"] = df["skill_cleaned"].map(lambda s: stem_text(s, porter))
    df["skill_lemmed"] = df["skill_cleaned"].map(lambda s: lemmatize_text(s, wn))

    # Giữ nguyên giá trị skill_type từ file CSV, không map lại
    if cols.type_name and cols.type_name in df.columns:
        df["skill_type"] = df[cols.type_name].astype(str)
    else:
        df["skill_type"] = "Hard Skill"

    # Abbreviation
    df["abbreviation"] = df[cols.name].map(lambda s: infer_abbreviation_from_name(s) if infer_abbr else "")

    unique_token_map: Dict[str, int] = {}
    if include_unique_token:
        unique_token_map = compute_unique_tokens(df["skill_cleaned"])
        def first_unique_token(cleaned: str) -> str:
            for tok in cleaned.split():
                if unique_token_map.get(tok, 0) == 1:
                    return tok
            return ""
        df["unique_token"] = df["skill_cleaned"].map(first_unique_token)

    df["match_on_stemmed"] = df["skill_len"].eq(1)

    df = df.sort_values(by=[cols.id], kind="mergesort")
    fields = ["skill_name", "skill_cleaned", "skill_type", "skill_lemmed", "skill_stemmed",
              "skill_len", "abbreviation", "match_on_stemmed"]
    if include_unique_token:
        fields.insert(7, "unique_token")

    records: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        d = {
            "skill_name": str(r[cols.name]),
            "skill_cleaned": str(r["skill_cleaned"]),
            "skill_type": str(r["skill_type"]),
            "skill_lemmed": str(r["skill_lemmed"]),
            "skill_stemmed": str(r["skill_stemmed"]),
            "skill_len": int(r["skill_len"]),
            "abbreviation": str(r["abbreviation"] or ""),
            "match_on_stemmed": bool(r["match_on_stemmed"]),
        }
        if include_unique_token:
            d["unique_token"] = str(r.get("unique_token", "") or "")
        records[str(r[cols.id])] = d

    return records

def main():
    # Cấu hình trực tiếp ở đây
    csv_path = "../data/raw/raw_skills_list.csv"
    json_out = "../data/processed/skills_processed.json"
    id_col = "id"
    name_col = "name"
    type_col = "type.name"
    include_unique_token = True
    infer_abbr = True
    indent = 2
    verbosity = 1

    setup_logging(verbosity)
    cols = Columns(id=id_col, name=name_col, type_name=type_col)

    logging.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    validate_columns(df, cols)

    logging.info("Processing %d records...", len(df))
    processed = build_processed(
        df,
        cols=cols,
        include_unique_token=include_unique_token,
        infer_abbr=infer_abbr,
    )

    logging.info("Writing JSON: %s", json_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=indent, sort_keys=False)

    n = len(processed)
    n_cert = sum(1 for v in processed.values() if v.get("skill_type") == "Certification")
    n_single = sum(1 for v in processed.values() if v.get("match_on_stemmed"))
    logging.info("Done. %d skills written. Certifications: %d. Single-token: %d.", n, n_cert, n_single)

if __name__ == "__main__":
    main()