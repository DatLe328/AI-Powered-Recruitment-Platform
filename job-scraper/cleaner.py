import re
def clean_job_title(text):
    text = text.strip()

    # Xoá "Mới" ở đầu (không phân biệt hoa/thường)
    text = re.sub(r'^\s*Mới\s+', '', text, flags=re.IGNORECASE)

    # Xoá "Gấp!!!", "Gấp!!", "Gấp!"… ở đầu
    text = re.sub(r'^\s*Gấp!+\s*', '', text, flags=re.IGNORECASE)

    # Xoá [ … ] hoặc [ … ] - ở đầu
    text = re.sub(r'^\s*\[.*?\]\s*-?\s*', '', text)

    # Nếu vẫn còn từ đầu tới dấu ! (trong trường hợp không phải "Gấp" nhưng vẫn có !)
    text = re.sub(r'^.*?!+\s*', '', text).strip()
    return text

def clean_link_in_text(text):
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text).strip()

    # Nếu còn dấu `-` mà sau đó chỉ có khoảng trắng (vì URL bị xoá) thì bỏ nốt
    text = re.sub(r'\s*-\s*$', '', text).strip()
    return text

def clean_front_punctuation_mark(text):
    text = re.sub(r'^[\s•\-–—]+', '', text)
    return text

import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk

from normalizer import Name_Normalizer

# khởi tạo
stop_words = set(stopwords.words("english"))
nn = Name_Normalizer()

out_path = "./output"
os.makedirs(out_path, exist_ok=True)

import re
def clean_punctuation(text):
    # Bỏ các dấu câu trừ / . -
    text = re.sub(r'[,:;!?\'\"]', ' ', text)
    return text
def clean_text(text, company):
    # loại tên công ty & chuẩn hoá
    company_norm = nn.normalize_company_name(company)
    if isinstance(company_norm, bytes):
        company_norm = company_norm.decode('utf-8', errors='ignore')
    text = text.replace(str(company), "").replace(str(company_norm), "")
    # lower
    text = text.lower()
    # bỏ stopwords
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def save_clean_descriptions(csv_file, output_name='ACCOUNTANT_DESCRIPTIONS.txt', mode='description'):
    df = pd.read_csv(csv_file)

    cleaned = []

    for _, row in df.iterrows():
        jd = str(row["description"])
        company = str(row["company"])

        text = clean_text(jd, company)


        if mode == "sentence":
            sentences = nltk.sent_tokenize(text)
            cleaned.extend(sentences)
        elif mode == "paragraph":
            sentences = nltk.sent_tokenize(text)
            for i in range(0, len(sentences), 3):
                para = " ".join(sentences[i:i+3])
                cleaned.append(para)
        else:  # description
            cleaned.append(text.replace('\n', ' '))  # <- đảm bảo 1 dòng
        print(jd)
        print(clean_punctuation(text))

    output_path = os.path.join(out_path, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line.strip() + " ")

    print(f"[✅] Cleaned descriptions saved to {output_path}")


if __name__ == "__main__":
    # chạy
    save_clean_descriptions("jobs.csv", mode="sentence")
