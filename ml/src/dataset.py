from datasets import load_dataset
import pandas as pd
import hashlib

label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

def download_dataset(url='cnamuangtoun/resume-job-description-fit'):
    dataset = load_dataset(url)
    df_train = dataset["train"].to_pandas()
    df_test = dataset['test'].to_pandas()
    return df_train, df_test

def map_label(df):
    df["label"] = df["label"].map(label_map)
    return df

def add_job_id(df):
    df['jd_id'] = pd.factorize(df['job_description_text'])[0]
    return df

def add_cv_id(
    df: pd.DataFrame,
    *,
    strategy: str = "hash",   # "hash" | "factorize" | "per_jd"
    prefix: str = "cv-"
) -> pd.DataFrame:
    """
    Gán cv_id cho mỗi resume.
    - strategy="hash": cv_id = "cv-" + sha1(resume_text)[:10]  (ổn định giữa train/test)
    - strategy="factorize": cv_id = "cv-" + index factorize theo toàn bộ df (chỉ ổn định trong 1 df)
    - strategy="per_jd": cv_id = f"jd{jd_id}-cv{rank}" (đánh số trong từng JD, phù hợp cho debug/hiển thị)
    """
    if "cv_id" in df.columns and df["cv_id"].notna().any():
        # đã có cv_id -> không ghi đè
        return df

    text = df["resume_text"].astype(str)

    if strategy == "hash":
        def _sha(s: str) -> str:
            return prefix + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
        df["cv_id"] = text.apply(_sha)

    elif strategy == "factorize":
        # chỉ ổn định trong phạm vi df hiện tại
        codes, _ = pd.factorize(text)
        df["cv_id"] = [f"{prefix}{i}" for i in codes]

    elif strategy == "per_jd":
        if "jd_id" not in df.columns:
            raise ValueError("per_jd requires 'jd_id' column. Call add_job_id() first.")
        # đánh số theo thứ tự xuất hiện trong từng JD
        df = df.copy()
        df["cv_id"] = (
            df.groupby("jd_id")
              .cumcount()
              .apply(lambda i: f"{prefix}{i:06d}")
        )
        # gắn JD prefix để tránh đụng key giữa các JD
        df["cv_id"] = "jd" + df["jd_id"].astype(str) + "-" + df["cv_id"]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'hash' | 'factorize' | 'per_jd'.")

    return df

def save_dataframe(df_train, df_test):
    df_train.to_csv('data/dataset_train.csv')
    df_test.to_csv('data/dataset_test.csv')

def get_from_huggingface():
    df_train, df_test = download_dataset()
    df_train = map_label(df_train)
    df_test = map_label(df_test)
    print(df_train)

    print('DF size:', df_train.shape[0])
    print('Number of Resumes:',df_train['resume_text'].nunique())
    print('Number of JD:',df_train['job_description_text'].nunique())

    df_train = add_job_id(df_train)
    df_test = add_job_id(df_test)
    print(df_train['jd_id'].max())
    print(df_test['jd_id'].max())
    print(df_test)

    save_dataframe(df_train, df_test) 

if __name__=='__main__':
    # df = pd.read_csv('../data/processed/dataset_balanced_train.csv')
    # df = map_label(df)
    # df = add_job_id(df)
    # df = add_cv_id(df)
    # df.to_csv('../data/train/dataset_balanced_train.csv', index=False)
    df = pd.read_csv('../data/raw/resume_job_matching_dataset.csv')
    df = add_job_id(df)
    df = add_cv_id(df)
    print(df['jd_id'].nunique())
    print(df['cv_id'].nunique())
    df.to_csv('../data/train/dataset_train.csv', index=False)