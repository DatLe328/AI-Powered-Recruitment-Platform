import os
import random
from typing import List, Tuple
import fitz
from skill_extractor import SkillExtractorFS
import json
import ftfy
import re

def read_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = text.replace("ï¼", " ")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = " ".join(text.split())
    return text

LOCATIONS = [
    "Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "Khánh Hòa", "Thừa Thiên Huế", "Quảng Ninh",
    "Vũng Tàu", "Nghệ An", "Thanh Hóa", "Hà Tĩnh", "Quảng Bình",
    "Bắc Ninh", "Bắc Giang", "Hưng Yên", "Hải Dương", "Nam Định",
    "Ninh Bình", "Phú Thọ", "Thái Nguyên", "Lào Cai", "Yên Bái",
    "Gia Lai", "Đắk Lắk", "Lâm Đồng", "Kon Tum", "Bình Thuận",
    "Long An", "Tiền Giang", "An Giang", "Kiên Giang", "Sóc Trăng",
    "Trà Vinh", "Vĩnh Long", "Bạc Liêu", "Cà Mau"
]
def process_to_json(folder_path: str, locations=None, seed: int | None = None):
    """
    Quét thư mục, đọc mọi PDF và trả về list[dict] có khóa:
      - resume_text: nội dung CV đã làm sạch
      - cv_location: chọn ngẫu nhiên từ 'locations'
    """
    if locations is None:
        locations = LOCATIONS

    if seed is not None:
        random.seed(seed)

    dataset = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, fname)
        try:
            raw_text = read_pdf_text(pdf_path)
            clean_text = preprocess_text(raw_text)
            if not clean_text:
                # bỏ qua file rỗng
                continue
            item = {
                "resume_text": clean_text,
                "cv_location": random.choice(locations),
            }
            dataset.append(item)
        except Exception as e:
            print(f"Lỗi xử lý {fname}: {e}")
    return dataset

def cv_sample():
    data = process_to_json("data/raw/job_description/INFORMATION-TECHNOLOGY", seed=42)
    save_dataset_json(data, "data/cvs_dataset.json")

def save_dataset_json(dataset, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def create_spacy_training_data_from_pdf(pdf_path: str, extractor: SkillExtractorFS) -> Tuple[str, dict]:

    raw_text = read_pdf_text(pdf_path)
    clean_text = preprocess_text(raw_text)
    annotation = extractor.annotate(clean_text)
    train_data = extractor.get_training_data_from_annotation(annotation)
    return train_data

def create_spacy_training_dataset_from_folder(folder_path: str) -> List[Tuple[str, dict]]:
    extractor = SkillExtractorFS()
    dataset = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, fname)
            train_data = create_spacy_training_data_from_pdf(pdf_path, extractor)
            dataset.append(train_data)
    return dataset

def save_spacy_data_as_doccano_jsonl(dataset: List[Tuple[str, dict]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for text, ann in dataset:
            item = {
                "text": text,
                "labels": [[start, end, label] for start, end, label in ann["entities"]]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__=='__main__':
    # dataset = create_spacy_training_dataset_from_folder('data/raw/job_description/INFORMATION-TECHNOLOGY')
    # save_spacy_data_as_doccano_jsonl(dataset, 'data/test/annotations.jsonl')
    cv_sample()