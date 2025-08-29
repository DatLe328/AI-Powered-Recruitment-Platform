import argparse, os, math, json
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, average_precision_score

LABEL_MAP = {"0": 0.0, "1": 0.5, "2": 1.0}

def read_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = ["resume_text", "job_description_text"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in CSV.")
    if "score" not in df.columns:
        if "label" not in df.columns:
            raise ValueError("CSV needs either 'score' or 'label'.")
        df["score"] = df["label"].map(LABEL_MAP).astype(float)
    df["resume_text"] = df["resume_text"].astype(str)
    df["job_description_text"] = df["job_description_text"].astype(str)
    df = df.dropna(subset=["resume_text", "job_description_text", "score"]).reset_index(drop=True)
    return df

def build_index(df: pd.DataFrame):
    cvs = sorted(set(df["resume_text"]))
    jds = sorted(set(df["job_description_text"]))
    cv2i = {t:i for i,t in enumerate(cvs)}
    jd2i = {t:i for i,t in enumerate(jds)}
    df = df.copy()
    df["cv_i"] = df["resume_text"].map(cv2i)
    df["jd_i"] = df["job_description_text"].map(jd2i)
    return cvs, jds, cv2i, jd2i, df

def embed_all(model: SentenceTransformer, texts, batch_size=128):
    return model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)

# ---- ranking metrics ----
def dcg_at_k(gains, k):
    s = 0.0
    for i, g in enumerate(gains[:k]):
        s += g / math.log2(i + 2.0)   # linear gain
    return s

def ndcg_at_k(grade_map, ranked, k):
    gains = [grade_map.get(i, 0.0) for i in ranked[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = sorted(grade_map.values(), reverse=True)
    idcg = dcg_at_k(ideal, min(k, len(ideal))) if ideal else 0.0
    return 0.0 if idcg == 0 else dcg / idcg

def recall_at_k(pos_set, ranked, k):
    if not pos_set: return None
    hits = sum(1 for i in ranked[:k] if i in pos_set)
    return hits / len(pos_set)

def mrr_at_k(pos_set, ranked, k):
    if not pos_set: return None
    for rank, i in enumerate(ranked[:k], start=1):
        if i in pos_set:
            return 1.0 / rank
    return 0.0

def pair_metrics(scores: np.ndarray, gold_score: np.ndarray):
    pearson = float(np.corrcoef(scores, gold_score)[0,1])
    spearman = float(pd.Series(scores).corr(pd.Series(gold_score), method="spearman"))
    # strict: chỉ Good Fit = 1 là dương
    y_strict = (gold_score >= 0.9).astype(int)
    # relaxed: Potential trở lên là dương
    y_relax  = (gold_score >= 0.5).astype(int)
    def safe_auc(y_true, y_score):
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return float('nan')
    def safe_ap(y_true, y_score):
        try:
            return float(average_precision_score(y_true, y_score))
        except Exception:
            return float('nan')
    return {
        "pearson": pearson,
        "spearman": spearman,
        "roc_auc_strict": safe_auc(y_strict, scores),
        "pr_auc_strict":  safe_ap(y_strict, scores),
        "roc_auc_relax":  safe_auc(y_relax,  scores),
        "pr_auc_relax":   safe_ap(y_relax,  scores),
    }

def evaluate_model(name, model_path, df, cvs, jds, cv2i, jd2i, k_list, out_dir=None):
    print(f"\n=== Evaluating: {name} ===")
    model = SentenceTransformer(model_path)

    # 1) embed unique CV/JD
    cv_emb = embed_all(model, cvs)
    jd_emb = embed_all(model, jds)

    # 2) pairwise scores for all rows
    cv_mat = cv_emb[df["cv_i"].values]
    jd_mat = jd_emb[df["jd_i"].values]
    scores_pair = np.sum(cv_mat * jd_mat, axis=1)  # cosine (đã normalize)
    pm = pair_metrics(scores_pair, df["score"].values)
    print(f" Pair metrics: "
          f"pearson={pm['pearson']:.4f}  spearman={pm['spearman']:.4f}  "
          f"ROC(A)=strict {pm['roc_auc_strict']:.4f} / relax {pm['roc_auc_relax']:.4f}  "
          f"PR(A)=strict {pm['pr_auc_strict']:.4f} / relax {pm['pr_auc_relax']:.4f}")

    # 3) build ground-truth per JD (graded + strict/relaxed)
    gt_grade = defaultdict(dict)   # jd_i -> {cv_i: grade in [0..1]}
    gt_pos_strict = defaultdict(set)
    gt_pos_relax  = defaultdict(set)
    for r in df.itertuples(index=False):
        grade = float(r.score)
        gt_grade[r.jd_i][r.cv_i] = max(grade, gt_grade[r.jd_i].get(r.cv_i, 0.0))
        if grade >= 0.9: gt_pos_strict[r.jd_i].add(r.cv_i)
        if grade >= 0.5: gt_pos_relax[r.jd_i].add(r.cv_i)

    # 4) retrieval metrics (per JD, average)
    ndcg_res, recS_res, recR_res, mrrS_res, mrrR_res = {k:[] for k in k_list}, {k:[] for k in k_list}, {k:[] for k in k_list}, {k:[] for k in k_list}, {k:[] for k in k_list}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        topk_rows = []

    for j_idx in range(len(jds)):
        q = jd_emb[j_idx]
        scores = cv_emb @ q
        order = np.argsort(scores)[::-1]  # best first

        for k in k_list:
            nd = ndcg_at_k(gt_grade[j_idx], order.tolist(), k)
            ndcg_res[k].append(nd)

            rs = recall_at_k(gt_pos_strict[j_idx], order.tolist(), k)
            rr = recall_at_k(gt_pos_relax[j_idx],  order.tolist(), k)
            if rs is not None: recS_res[k].append(rs)
            if rr is not None: recR_res[k].append(rr)

            ms = mrr_at_k(gt_pos_strict[j_idx], order.tolist(), k)
            mr = mrr_at_k(gt_pos_relax[j_idx],  order.tolist(), k)
            if ms is not None: mrrS_res[k].append(ms)
            if mr is not None: mrrR_res[k].append(mr)

        if out_dir:
            k_show = max(k_list)
            for rank, cv_i in enumerate(order[:k_show], start=1):
                topk_rows.append({
                    "model": name,
                    "jd_i": j_idx,
                    "cv_i": cv_i,
                    "rank": rank,
                    "score": float(scores[cv_i]),
                    "grade": float(gt_grade[j_idx].get(cv_i, 0.0)),
                    "jd_text": jds[j_idx][:200],
                    "cv_text": cvs[cv_i][:200]
                })

    # print summary
    for k in k_list:
        m_ndcg = float(np.mean(ndcg_res[k])) if ndcg_res[k] else float('nan')
        m_rs   = float(np.mean(recS_res[k])) if recS_res[k] else float('nan')
        m_rr   = float(np.mean(recR_res[k])) if recR_res[k] else float('nan')
        m_mrs  = float(np.mean(mrrS_res[k])) if mrrS_res[k] else float('nan')
        m_mrr  = float(np.mean(mrrR_res[k])) if mrrR_res[k] else float('nan')
        print(f" Retrieval@{k}:  NDCG={m_ndcg:.4f} | Recall(strict)={m_rs:.4f}  Recall(relax)={m_rr:.4f} "
              f"| MRR(strict)={m_mrs:.4f}  MRR(relax)={m_mrr:.4f}")

    if out_dir:
        pd.DataFrame(topk_rows).to_csv(os.path.join(out_dir, f"topk_{name}.csv"), index=False)

    return pm  # pair metrics (dict)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="../../data/processed/dataset_test.csv")
    ap.add_argument("--models", type=str,
                    default="sbert_finetuned_cv_jd,sentence-transformers/all-MiniLM-L6-v2",
                    help="comma-separated model paths/names")
    ap.add_argument("--k", type=int, nargs="+", default=[1,5,10])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    df = read_data(args.data_csv)
    df = df.iloc[:, 1:]
    df.drop('jd_id',axis=1, inplace=True)
    print(df.info())
    cvs, jds, cv2i, jd2i, df_idx = build_index(df)

    results = {}
    for i, mp in enumerate([m.strip() for m in args.models.split(",") if m.strip()]):
        name = f"m{i+1}"
        pm = evaluate_model(name, mp, df_idx, cvs, jds, cv2i, jd2i, args.k, out_dir=args.out_dir)
        results[name] = {"model_path": mp, **pm}

    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Summary (pair metrics) ===")
    for name, row in results.items():
        print(f"{name:>4} | {row['model_path']}")
        print(f"     pearson={row['pearson']:.4f} spearman={row['spearman']:.4f} "
              f"ROC(A) strict/relax={row['roc_auc_strict']:.4f}/{row['roc_auc_relax']:.4f} "
              f"PR(A)  strict/relax={row['pr_auc_strict']:.4f}/{row['pr_auc_relax']:.4f}")

if __name__ == "__main__":
    main()
