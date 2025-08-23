from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Sequence
from pathlib import Path
import json, re, hashlib
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import spearmanr, pearsonr
from ..features import build_features

DEFAULT_FEATURES: List[str] = [
    "vendor_cov", "bm25_full_norm", "bm25_skills_norm", "bm25_combo", "emb_cosine_norm"
]
DEFAULT_NDCG_AT: List[int] = [5, 10]


# ============= Helpers: IDs, labels, metrics =============
def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _stable_hash_id(text: str, prefix: str) -> str:
    h = hashlib.sha1(_norm_text(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}"

def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "jd_id" not in out.columns or out["jd_id"].isna().any():
        out["jd_id"] = out["job_description_text"].astype(str).apply(lambda s: _stable_hash_id(s, "jd-"))
    if "cv_id" not in out.columns or out["cv_id"].isna().any():
        out["cv_id"] = out["resume_text"].astype(str).apply(lambda s: _stable_hash_id(s, "cv-"))
    return out

def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s): return True
    try:
        pd.to_numeric(s); return True
    except Exception:
        return False

def _encode_labels(labels: pd.Series,
                   label_map: Optional[Dict[str, int]] = None) -> np.ndarray:
    if label_map is not None:
        y = labels.map(label_map)
        if y.isna().any():
            raise ValueError(f"Unmapped labels: {labels[y.isna()].unique().tolist()}")
        y = y.astype(int).to_numpy()
    else:
        if not _is_numeric_series(labels):
            raise ValueError("Non-numeric labels — please provide label_map for string labels.")
        y = pd.to_numeric(labels, errors="raise").to_numpy()
        y = np.rint(y).astype(int)
    shift = int(y.min())
    y0 = (y - shift).astype(int)
    return y0

def _group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby("jd_id").size().tolist()

def _ordinalize(y: np.ndarray) -> Tuple[np.ndarray, Dict[float, int]]:
    uniq = np.sort(np.unique(y))
    mapping = {v: i for i, v in enumerate(uniq)}
    y_ord = np.vectorize(mapping.get)(y).astype(int)
    return y_ord, mapping

def _gain_vector(K: int, mode: str = "exp") -> List[int]:
    if mode == "lin": return list(range(K))
    return [(2 ** i - 1) for i in range(K)]

def _ndcg_group(scores: np.ndarray, labels_ord: np.ndarray, k: int, gain: List[int]) -> float:
    order = np.argsort(scores)[::-1]
    gains = np.take(gain, labels_ord[order])
    disc  = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg   = float((gains[:k] * disc[:k]).sum())
    ideal = np.sort(labels_ord)[::-1]
    gains_ideal = np.take(gain, ideal)
    idcg  = float((gains_ideal[:k] * disc[:k]).sum())
    return dcg / (idcg + 1e-12)

def _eval_per_jd(df: pd.DataFrame, score_col: str, label_col: str,
                 ks: Sequence[int] = (5,10), gain_mode: str = "exp") -> Dict[str, float]:
    pears = []; spear = []; sizes = []
    ndcg_buckets = {f"ndcg@{k}": [] for k in ks}
    for _, g in df.groupby("jd_id"):
        if len(g) < 2: continue
        y = g[label_col].to_numpy(int)
        s = g[score_col].to_numpy(float)
        try: pr = pearsonr(s, y).statistic
        except Exception: pr = np.nan
        try: sr = spearmanr(s, y).statistic
        except Exception: sr = np.nan
        pears.append(pr); spear.append(sr); sizes.append(len(g))
        y_ord, _ = _ordinalize(y); gain = _gain_vector(int(y_ord.max()) + 1, mode=gain_mode)
        for k in ks:
            ndcg_buckets[f"ndcg@{k}"].append(_ndcg_group(s, y_ord, k=k, gain=gain))
    out = {}
    if sizes:
        w = np.array(sizes, float)
        def wmean(a): 
            a = np.array(a, float); m = ~np.isnan(a)
            return float((a[m]*w[m]).sum() / (w[m].sum() + 1e-12)) if m.any() else 0.0
        out["pearson_w"] = wmean(pears)
        out["spearman_w"] = wmean(spear)
        for k in ks: out[f"ndcg@{k}"] = wmean(ndcg_buckets[f"ndcg@{k}"])
    else:
        out["pearson_w"] = out["spearman_w"] = 0.0
        for k in ks: out[f"ndcg@{k}"] = 0.0
    return {k: round(v, 4) for k, v in out.items()}

def _monotone_constraints(features: List[str], positive: bool = True) -> str:
    sign = "1" if positive else "-1"
    return "(" + ",".join([sign] * len(features)) + ")"


# ==================== Configs ====================
@dataclass
class FeatureConfig:
    fuzzy_vendor: float = 0.70
    max_alias_len: int = 4
    laplace_a: float = 1.0
    laplace_b: float = 1.0
    bm25_full_weight: float = 0.5
    bm25_skills_weight: float = 0.5
    final_skill_weight: float = 0.5
    final_bm25_weight: float = 0.5

    use_embedding: bool = True
    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_device: str | None = None
    emb_batch_size: int = 64
    emb_cache_path: str | None = "cache/emb_sbert.pkl"
    emb_per_jd_norm: bool = True

@dataclass
class RankerConfig:
    features: List[str] = None
    ndcg_at: List[int] = None
    label_map: Optional[Dict[str, int]] = None
    gain_mode: str = "exp"               # chỉ dùng khi tự tính metric báo cáo

    # XGBRanker params
    objective: str = "rank:ndcg"         # 'rank:pairwise' / 'rank:ndcg' / 'rank:map'
    learning_rate: float = 0.05
    n_estimators: int = 1500
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    tree_method: str = "hist"            # 'hist' | 'gpu_hist'
    monotone_positive: bool = True
    verbose: int = 1

    def __post_init__(self):
        if self.features is None:
            self.features = DEFAULT_FEATURES
        if self.ndcg_at is None:
            self.ndcg_at = DEFAULT_NDCG_AT


# ==================== Main class ====================
class CVJDXGBRanker:
    def __init__(self, feat_cfg: FeatureConfig, rank_cfg: Optional[RankerConfig] = None, random_state: int = 42):
        self.feat_cfg = feat_cfg
        self.rank_cfg = rank_cfg or RankerConfig()
        self.random_state = int(random_state)
        self.model: Optional[XGBRanker] = None
        self.feature_names_: List[str] = self.rank_cfg.features

    # ---- Build features (truyền thêm tham số embedding nếu cần) ----
    def build_features(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        need = {"job_description_text","resume_text"}
        miss = need - set(df_pairs.columns)
        if miss:
            raise ValueError(f"build_features(): missing columns {miss}")
        df_pairs = _ensure_ids(df_pairs)
        feats = build_features(
            df=df_pairs[["jd_id","job_description_text","cv_id","resume_text"]].copy(),
            fuzzy_vendor=self.feat_cfg.fuzzy_vendor,
            max_alias_len=self.feat_cfg.max_alias_len,
            laplace_a=self.feat_cfg.laplace_a,
            laplace_b=self.feat_cfg.laplace_b,
            bm25_full_weight=self.feat_cfg.bm25_full_weight,
            bm25_skills_weight=self.feat_cfg.bm25_skills_weight,
            final_skill_weight=self.feat_cfg.final_skill_weight,
            final_bm25_weight=self.feat_cfg.final_bm25_weight,
            use_embedding=self.feat_cfg.use_embedding,
            emb_model_name=self.feat_cfg.emb_model_name,
            emb_device=self.feat_cfg.emb_device,
            emb_batch_size=self.feat_cfg.emb_batch_size,
            emb_cache_path=self.feat_cfg.emb_cache_path,
            emb_per_jd_norm=self.feat_cfg.emb_per_jd_norm,
        )
        missing = [c for c in self.feature_names_ if c not in feats.columns]
        if missing:
            raise ValueError(f"Required features not built: {missing}")
        return feats

    # ---- Fit ----
    def fit(self, df_labeled: pd.DataFrame, val_size: float = 0.2) -> Dict[str, float]:
        need = {"job_description_text","resume_text","label"}
        miss = need - set(df_labeled.columns)
        if miss:
            raise ValueError(f"fit(): missing columns {miss}")

        df = _ensure_ids(df_labeled.copy())

        # encode labels
        y_int = _encode_labels(df["label"], self.rank_cfg.label_map)
        df["label_int"] = y_int

        # keep groups with >=2 and variance > 0
        counts = df["jd_id"].value_counts()
        keep = set(counts[counts >= 2].index)
        df = df[df["jd_id"].isin(keep)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                "All groups have size 1 — XGBRanker requires ≥2 CV per JD. "
                "Hãy dựng negative sampling (slate) cho mỗi JD, hoặc dùng regressor."
            )
        var_by_group = df.groupby("jd_id")["label_int"].agg(lambda x: int(x.max() - x.min()))
        keep2 = set(var_by_group[var_by_group > 0].index)
        df = df[df["jd_id"].isin(keep2)].reset_index(drop=True)
        if df.empty:
            raise ValueError("No label variance within groups; cannot train a ranker.")

        # features
        feats = self.build_features(df)
        data = feats.merge(df[["jd_id","cv_id","label_int"]], on=["jd_id","cv_id"], how="left")

        # split theo group
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=self.random_state)
        idx_tr, idx_va = next(gss.split(np.arange(len(data)), groups=data["jd_id"].values))
        tr, va = data.iloc[idx_tr].reset_index(drop=True), data.iloc[idx_va].reset_index(drop=True)

        X_tr = tr[self.feature_names_]
        y_tr = tr["label_int"].to_numpy(int)
        grp_tr = _group_sizes(tr)

        X_va = va[self.feature_names_]
        y_va = va["label_int"].to_numpy(int)
        grp_va = _group_sizes(va)

        mono = _monotone_constraints(self.feature_names_, positive=self.rank_cfg.monotone_positive)
        self.model = XGBRanker(
            objective=self.rank_cfg.objective,       # "rank:ndcg" khuyến nghị
            eval_metric=[f"ndcg@{k}" for k in self.rank_cfg.ndcg_at],
            learning_rate=self.rank_cfg.learning_rate,
            n_estimators=self.rank_cfg.n_estimators,
            max_depth=self.rank_cfg.max_depth,
            subsample=self.rank_cfg.subsample,
            colsample_bytree=self.rank_cfg.colsample_bytree,
            reg_lambda=self.rank_cfg.reg_lambda,
            tree_method=self.rank_cfg.tree_method,
            monotone_constraints=mono,
            random_state=self.random_state,
        )
        self.model.fit(
            X_tr, y_tr,
            group=grp_tr,
            eval_set=[(X_va, y_va)],
            eval_group=[grp_va],
            verbose=self.rank_cfg.verbose,
        )

        # ngoại kiểm theo JD
        va = va.copy()
        va["pred"] = self.model.predict(X_va)
        metrics = _eval_per_jd(va, score_col="pred", label_col="label_int",
                               ks=self.rank_cfg.ndcg_at, gain_mode=self.rank_cfg.gain_mode)
        return metrics

    # ---- Predict / Rank ----
    def predict(self, df_pairs: pd.DataFrame, return_features: bool = False) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("No trained/loaded model.")
        feats = self.build_features(df_pairs)
        X = feats[self.feature_names_]
        scores = self.model.predict(X)
        out = feats[["jd_id","cv_id"]].copy()
        out["pred"] = scores.astype(float)
        if return_features:
            for c in self.feature_names_:
                out[c] = feats[c].values
        return out

    def rank(self, df_pairs: pd.DataFrame, topk: Optional[int] = None) -> pd.DataFrame:
        preds = self.predict(df_pairs, return_features=False)
        rows = []
        for gid, g in preds.groupby("jd_id"):
            if len(g) < 2:
                continue
            g2 = g.sort_values("pred", ascending=False)
            if topk is not None:
                g2 = g2.head(int(topk))
            rows.append(g2)
        return pd.concat(rows, axis=0).reset_index(drop=True)

    # ---- Save / Load ----
    def save(self, model_path: str, config_path: Optional[str] = None):
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        if config_path:
            cfg = {
                "feat_cfg": asdict(self.feat_cfg),
                "rank_cfg": asdict(self.rank_cfg),
                "feature_names_": self.feature_names_,
                "random_state": self.random_state,
            }
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_path: str, config_path: str) -> "CVJDXGBRanker":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        feat_cfg = FeatureConfig(**cfg["feat_cfg"])
        rank_cfg = RankerConfig(**{k: v for k, v in cfg["rank_cfg"].items() if k in RankerConfig().__dict__})
        obj = cls(feat_cfg=feat_cfg, rank_cfg=rank_cfg, random_state=cfg.get("random_state", 42))
        obj.feature_names_ = cfg.get("feature_names_", DEFAULT_FEATURES)

        mono = _monotone_constraints(obj.feature_names_, positive=obj.rank_cfg.monotone_positive)
        obj.model = XGBRanker(
            objective=obj.rank_cfg.objective,
            eval_metric=[f"ndcg@{k}" for k in obj.rank_cfg.ndcg_at],
            learning_rate=obj.rank_cfg.learning_rate,
            n_estimators=obj.rank_cfg.n_estimators,
            max_depth=obj.rank_cfg.max_depth,
            subsample=obj.rank_cfg.subsample,
            colsample_bytree=obj.rank_cfg.colsample_bytree,
            reg_lambda=obj.rank_cfg.reg_lambda,
            tree_method=obj.rank_cfg.tree_method,
            monotone_constraints=mono,
            random_state=obj.random_state,
        )
        obj.model.load_model(model_path)
        return obj
