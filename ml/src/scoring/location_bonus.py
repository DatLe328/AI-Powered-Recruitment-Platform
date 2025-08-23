from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Iterable
import re, unicodedata

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def _norm(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s

@dataclass
class LocationConfig:
    alpha: float = 0.10
    remote_keywords: Iterable[str] = ("remote", "work from home", "wfh", "fully remote")
    hybrid_keywords: Iterable[str] = ("hybrid",)
    onsite_keywords: Iterable[str] = ("onsite", "on-site")
    neutral_when_cv_missing: float = 0.85
    base_min_score: float = 0.60

class LocationScorer:
    def __init__(self, cfg: Optional[LocationConfig] = None):
        self.cfg = cfg or LocationConfig()
        self._kw_remote = tuple(_norm(k) for k in self.cfg.remote_keywords)
        self._kw_hybrid = tuple(_norm(k) for k in self.cfg.hybrid_keywords)
        self._kw_onsite = tuple(_norm(k) for k in self.cfg.onsite_keywords)

    # --------- public API ----------
    def score_pair(self, jd_location: Optional[str], cv_location: Optional[str], jd_text: Optional[str] = None) -> float:
        """
        Trả về loc_score ∈ [0,1].
        - Nếu JD là remote → 1.0
        - Nếu thiếu jd_location: coi trung lập (1.0)
        - Nếu thiếu cv_location: neutral (cfg.neutral_when_cv_missing)
        - Onsite/hybrid: same city 1.0 > same province 0.9 > same country 0.8 > khác quốc gia 0.6
        """
        # Nếu JD nhắc tới remote trong mô tả
        if self._is_remote_text(jd_text) or self._is_remote_text(jd_location):
            return 1.0

        # Không có location của JD → trung lập (không phạt)
        if not jd_location or not _norm(jd_location):
            return 1.0

        # Không có location của CV → trung lập nhẹ
        if not cv_location or not _norm(cv_location):
            return float(self.cfg.neutral_when_cv_missing)

        jd_loc = self._parse_loc(jd_location)
        cv_loc = self._parse_loc(cv_location)

        # onsite/hybrid → xét mức độ gần
        if self._is_onsite_or_hybrid_text(jd_text) or self._is_onsite_or_hybrid_text(jd_location):
            return self._proximity_score(jd_loc, cv_loc)

        # nếu không rõ (mặc định xem nhẹ) → dùng proximity nhưng cắt trên
        return self._proximity_score(jd_loc, cv_loc)

    def apply_bonus_dataframe(self, df, jd_loc_col: str = "job_location", cv_loc_col: str = "cv_location",
                              base_score_col: str = "pred", out_col: str = "final_with_loc",
                              jd_text_col: str = "job_description_text", blend: bool = True, tie_break: bool = False):
        """
        Thêm cột loc_score và:
        - blend=True  → final = (1-α)*base + α*loc_score
        - tie_break=True → chỉ sắp xếp lại theo loc_score làm khóa phụ, không tạo final
        """
        import pandas as pd
        df = df.copy()
        loc_scores = []
        for r in df.itertuples(index=False):
            jd_loc = getattr(r, jd_loc_col) if jd_loc_col in df.columns else None
            cv_loc = getattr(r, cv_loc_col) if cv_loc_col in df.columns else None
            jd_text = getattr(r, jd_text_col) if jd_text_col in df.columns else None
            loc_scores.append(self.score_pair(jd_loc, cv_loc, jd_text))
        df["loc_score"] = loc_scores

        if blend:
            alpha = float(self.cfg.alpha)
            df[out_col] = (1 - alpha) * df[base_score_col].astype(float) + alpha * df["loc_score"].astype(float)
        if tie_break:
            # sort in-place suggestion: group by jd_id if có
            by = ["jd_id", base_score_col, "loc_score"] if "jd_id" in df.columns else [base_score_col, "loc_score"]
            asc = [True, False, False] if "jd_id" in df.columns else [False, False]
            df = df.sort_values(by=by, ascending=asc, kind="mergesort")  # mergesort để ổn định
        return df

    # --------- internals ----------
    def _is_remote_text(self, text: Optional[str]) -> bool:
        t = _norm(text)
        return any(k in t for k in self._kw_remote)

    def _is_onsite_or_hybrid_text(self, text: Optional[str]) -> bool:
        t = _norm(text)
        return any(k in t for k in self._kw_onsite + self._kw_hybrid)

    def _parse_loc(self, s: str) -> Dict[str, str]:
        """
        Rất đơn giản: tách bởi dấu phẩy, lấy city, province, country theo thứ tự từ trái qua phải.
        Ví dụ: 'District 1, Ho Chi Minh City, Vietnam' → {'city':'district 1','province':'ho chi minh city','country':'vietnam'}
        """
        t = _norm(s)
        parts = [p.strip() for p in t.split(",") if p.strip()]
        city = parts[0] if len(parts) >= 1 else ""
        province = parts[1] if len(parts) >= 2 else ""
        country = parts[-1] if parts else ""
        # cleanup: bỏ từ chung
        for w in ("city", "province", "state", "district"):
            city = re.sub(rf"\b{w}\b", "", city).strip()
            province = re.sub(rf"\b{w}\b", "", province).strip()
        return {"city": city, "province": province, "country": country}

    def _proximity_score(self, jd_loc: Dict[str,str], cv_loc: Dict[str,str]) -> float:
        # same city
        if jd_loc["city"] and jd_loc["city"] == cv_loc["city"]:
            return 1.00
        # same province
        if jd_loc["province"] and jd_loc["province"] == cv_loc["province"]:
            return 0.90
        # same country
        if jd_loc["country"] and jd_loc["country"] == cv_loc["country"]:
            return 0.80
        # different / unknown granularity
        return float(self.cfg.base_min_score)
