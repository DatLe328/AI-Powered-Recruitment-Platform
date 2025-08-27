import pandas as pd
from torch.utils.data import DataLoader, random_split
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

# 1. Load dữ liệu
df = pd.read_csv("../../data/processed/dataset_balanced_train.csv")

label_map = {"No Fit": 0.0, "Potential Fit": 0.5, "Good Fit": 1.0}
df["score"] = df["label"].map(label_map)

# Tạo dataset InputExample
examples = [InputExample(texts=[row.resume_text, row.job_description_text], label=row.score)
            for row in df.itertuples()]

# Chia dữ liệu train/val/test
train_size = int(0.8 * len(examples))
val_size = int(0.1 * len(examples))
test_size = len(examples) - train_size - val_size
train_data, val_data, test_data = random_split(examples, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=16)

# Load model (English, chính xác cao)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Loss
train_loss = losses.CosineSimilarityLoss(model)

# Evaluation (dùng Pearson & Spearman correlation)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_data)

# Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,                 # 3–5 là hợp lý
    warmup_steps=90,          # 10% tổng step
    evaluation_steps=50,      # validate sau mỗi 50 step
    output_path="sbert_finetuned_cv_jd"
)