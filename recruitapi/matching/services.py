# matching/services.py
from django.db import connection
from ml.embeddings import embed_texts
from ml.vectorstore.faiss_store import is_loaded as faiss_is_loaded, add as faiss_add

SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cv_embeddings (
  cv_id      BIGINT PRIMARY KEY,
  dim        SMALLINT NOT NULL,
  vec        BYTEA NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


SQL_UPSERT_EMB = """
INSERT INTO cv_embeddings (cv_id, dim, vec, updated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (cv_id) DO UPDATE SET
  dim = EXCLUDED.dim,
  vec = EXCLUDED.vec,
  updated_at = NOW();
"""


SQL_DELETE_EMB = "DELETE FROM cv_embeddings WHERE cv_id = %s"


def ensure_cv_embedding_table():
    with connection.cursor() as cur:
        cur.execute(SQL_CREATE_TABLE)

def upsert_cv_embedding_and_update_faiss(cv_id: int | str, resume_text: str):
    ensure_cv_embedding_table()
    emb = embed_texts([resume_text])[0]
    dim = emb.shape[0]
    with connection.cursor() as cur:
        cur.execute(SQL_UPSERT_EMB, [cv_id, dim, memoryview(emb.tobytes())])
    if faiss_is_loaded():
        faiss_add(emb.reshape(1, -1), [cv_id])

def delete_cv_embedding(cv_id: int | str):
    ensure_cv_embedding_table()
    with connection.cursor() as cur:
        cur.execute(SQL_DELETE_EMB, [cv_id])