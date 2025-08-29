import math
import numpy as np
from django.core.management.base import BaseCommand
from matching.models import CV
from ml.embeddings import embed_texts
from ml.vectorstore import faiss_store

class Command(BaseCommand):
    help = "Rebuild FAISS index từ toàn bộ CV.active"

    def add_arguments(self, parser):
        parser.add_argument("--batch", type=int, default=256, help="batch embed")
        parser.add_argument("--hnsw", action="store_true", help="dùng HNSW thay vì Flat/IVF")
        parser.add_argument("--ivf", type=int, default=0, help="dùng IVF với nlist (0 = tắt)")
        parser.add_argument("--save-to", type=str, default="", help="đường dẫn file index (nếu module cho phép)")

    def handle(self, *args, **opts):
        qs = CV.objects.filter(is_active=True).only("id", "resume_text").order_by("id")
        ids, texts = [], []
        for cv in qs.iterator(chunk_size=2000):
            if cv.resume_text:
                ids.append(cv.id)
                texts.append(cv.resume_text)

        self.stdout.write(f"Embedding {len(ids)} CV ...")
        # embed theo batch để không tràn RAM/GPU
        embs = []
        B = opts["batch"]
        for i in range(0, len(texts), B):
            embs.append(embed_texts(texts[i:i+B]))
        X = np.vstack(embs).astype("float32")   # shape [N, d]
        I = np.array(ids, dtype="int64")

        # build index bằng helper trong module faiss_store (tận dụng save/load sẵn có)
        # Nếu module của bạn có API build/swap, dùng trực tiếp; nếu không, dùng create/update:
        if opts["ivf"] > 0:
            index = faiss_store.build_ivf(X, I, nlist=opts["ivf"])   # ví dụ: hàm tiện ích của bạn
        elif opts["hnsw"]:
            index = faiss_store.build_hnsw(X, I)                     # ví dụ: hàm tiện ích của bạn
        else:
            index = faiss_store.build_flat(X, I)                     # ví dụ: hàm tiện ích của bạn

        saved = faiss_store.save(index, path=opts["save_to"] or None)  # nếu module hỗ trợ
        self.stdout.write(self.style.SUCCESS(f"Rebuild xong ({len(ids)} vectors). Save={bool(saved)}"))
