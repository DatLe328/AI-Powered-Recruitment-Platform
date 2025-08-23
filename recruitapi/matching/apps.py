from django.apps import AppConfig
import os, threading, logging
log = logging.getLogger(__name__)

class MatchingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'matching'

    # def ready(self):
    #     def _warm():
    #         try:
    #             from ml.ai_gateway import load_model, rank_cv_for_jd
    #             load_model()

    #             _ = rank_cv_for_jd(
    #                 job_requirement="python fastapi postgresql docker",
    #                 job_description="ci/cd monitoring",
    #                 candidates=[
    #                     {"cv_id":"warm-1","resume_text":"Python FastAPI PostgreSQL Docker"},
    #                     {"cv_id":"warm-2","resume_text":"Java Spring MySQL Docker"}
    #                 ],
    #                 topk=1,
    #             )
    #         except Exception as e:
    #             import logging; logging.getLogger(__name__).warning(f"AI warmup skipped: {e}")
    #     import threading; threading.Thread(target=_warm, daemon=True).start()