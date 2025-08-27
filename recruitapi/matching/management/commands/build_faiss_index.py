from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import connection
import numpy as np
from tqdm import tqdm

from ml.embeddings import embed_texts
from ml.vectorstore.faiss_store import build_new

SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cv_embeddings (
  cv_id      BIGINT PRIMARY KEY,
  dim        SMALLINT NOT NULL,
  vec        BYTEA NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

SQL_SELECT_CANDIDATES = lambda: getattr(settings, "CANDIDATE_SQL", "SELECT id, resume_text FROM candidates")

SQL_UPSERT_EMB = """
INSERT INTO cv_embeddings (cv_id, dim, vec, updated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (cv_id) DO UPDATE SET
  dim = EXCLUDED.dim,
  vec = EXCLUDED.vec,
  updated_at = NOW();
"""

SQL_SELECT_ALL_EMB = "SELECT cv_id, dim, vec FROM cv_embeddings"

class Command(BaseCommand):
    help = "Build FAISS index từ dữ liệu CV lưu trong PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--index-kind", type=str, default="hnsw", choices=["hnsw", "flat"])
        parser.add_argument("--recompute", action="store_true", help="Tính lại embedding cho tất cả CV (không dùng cache)")

    def handle(self, *args, **opts):
        batch_size = opts["batch_size"]
        index_kind = opts["index_kind"]
        recompute = opts["recompute"]

        with connection.cursor() as cur:
            # 1) Tạo bảng nếu chưa có
            cur.execute(SQL_CREATE_TABLE)

            # 2) Lấy dữ liệu CV (id, text)
            cur.execute(SQL_SELECT_CANDIDATES())
            rows = cur.fetchall()
            if not rows:
                self.stdout.write(self.style.WARNING("Không có CV nào từ CANDIDATE_SQL."))
                return

            cv_ids, texts = zip(*rows)
            cv_ids = list(cv_ids); texts = list(texts)

            # 3) Với --recompute: encode tất cả và upsert
            if recompute:
                self.stdout.write("Tạo/ghi lại embedding vào PostgreSQL...")
                for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                    chunk_texts = texts[i:i+batch_size]
                    chunk_ids = cv_ids[i:i+batch_size]
                    embs = embed_texts(chunk_texts)  # (B, D) float32
                    dim = embs.shape[1]
                    for cid, v in zip(chunk_ids, embs):
                        cur.execute(SQL_UPSERT_EMB, [cid, dim, memoryview(v.tobytes())])

            # 4) Đọc toàn bộ embedding để build FAISS
            self.stdout.write("Đọc embedding từ PostgreSQL...")
            cur.execute(SQL_SELECT_ALL_EMB)
            rows = cur.fetchall()
            if not rows:
                self.stdout.write(self.style.WARNING("Bảng cv_embeddings trống. Hãy chạy lại với --recompute."))
                return

            ids = []
            vecs = []
            for cid, dim, blob in tqdm(rows, desc="Load"):
                v = np.frombuffer(bytes(blob), dtype=np.float32).reshape(dim)
                ids.append(cid)
                vecs.append(v)
            embs = np.vstack(vecs)

        # 5) Build & persist FAISS
        build_new(embs, ids, kind=index_kind, meta={"source": "postgres"})
        self.stdout.write(self.style.SUCCESS(f"Đã build FAISS index cho {len(ids)} CV."))
