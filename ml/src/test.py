# test_with_location_bonus.py
import pandas as pd
from models import CVJDXGBRanker           # hoặc CVJDXGBScorer nếu bạn dùng scorer
from scoring import LocationScorer, LocationConfig

# 1) Load model
ranker = CVJDXGBRanker.load("models/xgb_ranker.json", "models/xgb_ranker_config.json")

# 2) 1 JD + 5 CV (thêm 2 cột location tùy chọn)
JD_TEXT = """We are hiring a Backend Engineer (onsite)."""
JD_LOC  = "District 1, Ho Chi Minh City, Vietnam"

rows = [
    {"jd_id":"demo-001","job_description_text":JD_TEXT,"job_location":JD_LOC,
     "cv_id":"cv-1","resume_text":"Python, FastAPI, PostgreSQL, Docker, AWS ECS","cv_location":"Thu Duc, Ho Chi Minh City, Vietnam"},
    {"jd_id":"demo-001","job_description_text":JD_TEXT,"job_location":JD_LOC,
     "cv_id":"cv-2","resume_text":"Java Spring Boot, MySQL, Docker","cv_location":"Hanoi, Vietnam"},
    {"jd_id":"demo-001","job_description_text":JD_TEXT,"job_location":JD_LOC,
     "cv_id":"cv-3","resume_text":"Data engineer: Python, Kafka, PostgreSQL, Kubernetes","cv_location":"Singapore"},
    {"jd_id":"demo-001","job_description_text":JD_TEXT,"job_location":JD_LOC,
     "cv_id":"cv-4","resume_text":"Frontend React, TypeScript","cv_location":"Ho Chi Minh City, Vietnam"},
    {"jd_id":"demo-001","job_description_text":JD_TEXT,"job_location":JD_LOC,
     "cv_id":"cv-5","resume_text":"DevOps: Docker, GitHub Actions, Terraform","cv_location":None},  # thiếu location
]
df_pairs = pd.DataFrame(rows)

# 3) Predict bằng model (điểm 'pred' theo nội dung)
preds = ranker.predict(df_pairs, return_features=False)

# 4) Áp điểm thưởng location (5–10% tổng điểm)
loc = LocationScorer(LocationConfig(alpha=0.10))
with_bonus = loc.apply_bonus_dataframe(
    preds.merge(df_pairs[["jd_id","cv_id","job_location","cv_location","job_description_text"]], on=["jd_id","cv_id"], how="left"),
    jd_loc_col="job_location",
    cv_loc_col="cv_location",
    base_score_col="pred",
    out_col="final_with_loc",
    blend=True,       # pha vào điểm
    tie_break=False   # hoặc True nếu chỉ muốn tie-break
)

# 5) Xếp hạng theo điểm cuối
ranked = with_bonus.sort_values(["jd_id","final_with_loc"], ascending=[True,False]).reset_index(drop=True)
print(ranked[["jd_id","cv_id","pred","loc_score","final_with_loc"]])

# # test_ranker_predict.py
# import pandas as pd
# from models import CVJDXGBRanker

# # 1) Load model & config đã train/saved trước đó
# MODEL_PATH  = "models/xgb_ranker.json"
# CONFIG_PATH = "models/xgb_ranker_config.json"
# ranker = CVJDXGBRanker.load(MODEL_PATH, CONFIG_PATH)

# # 2) Tạo 1 JD + 5 CV để test
# JD_TEXT = """We are hiring a Backend Engineer.
# Must-have: Python, FastAPI, PostgreSQL, Docker, CI/CD on GitHub Actions.
# Nice-to-have: AWS (ECS/ECR), Kafka, monitoring (Prometheus/Grafana)."""

# CANDIDATES = [
#     ("cv-1", """Python developer with FastAPI, PostgreSQL, Docker; deployed on AWS ECS; built CI/CD pipelines."""),
#     ("cv-2", """Java Spring Boot, MySQL, some Docker; learning Python; basic CI/CD."""),
#     ("cv-3", """Data engineer: Python, Airflow, Kafka; strong PostgreSQL; Docker and Kubernetes; Grafana."""),
#     ("cv-4", """Frontend React/Next.js, TypeScript; a bit of Node.js; no Docker experience."""),
#     ("cv-5", """DevOps: Docker, Kubernetes, GitHub Actions, Terraform; some Python; AWS ECR/ECS; monitoring with Prometheus."""),
# ]


# rows = []
# for cv_id, cv_text in CANDIDATES:
#     rows.append({
#         "jd_id": "demo-001",
#         "job_description_text": JD_TEXT,
#         "cv_id": cv_id,
#         "resume_text": cv_text
#     })
# df_pairs = pd.DataFrame(rows)

# # 3) Dự đoán & xếp hạng (giảm dần theo "pred")
# ranked = ranker.rank(df_pairs, topk=None)
# print(ranked[["jd_id","cv_id","pred"]])
