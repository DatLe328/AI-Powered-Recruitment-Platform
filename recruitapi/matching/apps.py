from django.apps import AppConfig
import os, threading, logging, sys
log = logging.getLogger(__name__)

class MatchingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'matching'

    def ready(self):
        mgmt_cmds_to_skip = {
            "makemigrations","migrate","collectstatic","shell",
            "createsuperuser","loaddata","dumpdata","test","pytest", "build_faiss_index"
        }
        if any(cmd in sys.argv for cmd in mgmt_cmds_to_skip):
            return

        def _warm():
            from ml.src.embedder import SbertConfig, get_sbert_embedder
            from django.conf import settings
            from pathlib import Path
            try:
                cfg = SbertConfig(
                    model_name=getattr(settings, "EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                    device=getattr(settings, "EMB_DEVICE", None),
                    batch_size=int(getattr(settings, "EMB_BATCH", 64)),
                    normalize=True,
                    cache_path=str(Path(__file__).resolve().parent.parent / "ml" / "src" / "cache" / "emb_sbert.pkl"),
                )
                get_sbert_embedder(cfg)
            except Exception as e:
                log.info(f"SBERT warm-up skipped: {e}")
            try:
                from ml.apis import load_model, rank_cv_for_jd
                load_model()

                _ = rank_cv_for_jd(
                    job_requirement="python fastapi postgresql docker",
                    job_description="ci/cd monitoring",
                    candidates=[
                        {"cv_id":"warm-1","resume_text":"Python FastAPI PostgreSQL Docker"},
                        {"cv_id":"warm-2","resume_text":"Java Spring MySQL Docker"}
                    ],
                    topk=1,
                )
                print("AI warmup completed successfully.")
                from ml.vectorstore.faiss_store import load as faiss_load
                ok = faiss_load()
                if ok:
                    log.info("FAISS index loaded.")
                else:
                    log.warning("FAISS index chưa có. Hãy chạy: python manage.py build_faiss_index --recompute")
            except Exception as e:
                import logging; logging.getLogger(__name__).warning(f"AI warmup skipped: {e}")