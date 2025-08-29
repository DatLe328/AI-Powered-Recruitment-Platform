import pandas as pd
from models import CVJDXGBRanker, FeatureConfig, RankerConfig

df = pd.read_csv("../data/train/dataset_balanced_train.csv")

feat_cfg = FeatureConfig(
    fuzzy_vendor=0.70,
    max_alias_len=4,
    bm25_full_weight=0.5,
    bm25_skills_weight=0.5,
    final_skill_weight=0.5,
    final_bm25_weight=0.5,

    use_embedding=True,
    emb_model_name="sentence-transformers/all-MiniLM-L6-v2",
    emb_device=None,
    emb_batch_size=64,
    emb_cache_path="cache/emb_sbert.pkl",
    emb_per_jd_norm=True, 
)

# 3) Cấu hình ranker
rank_cfg = RankerConfig(
    features=["vendor_cov","bm25_full_norm","bm25_skills_norm","bm25_combo","emb_cosine_norm"],
    objective="rank:ndcg",
    ndcg_at=[5, 10],
    learning_rate=0.05,
    n_estimators=1500,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    monotone_positive=True,
    verbose=1,
)

ranker = CVJDXGBRanker(feat_cfg, rank_cfg, random_state=42)
metrics = ranker.fit(df, val_size=0.2)
print("Val metrics:", metrics)

ranker.save("../models/xgb_ranker.json", "../models/xgb_ranker_config.json")