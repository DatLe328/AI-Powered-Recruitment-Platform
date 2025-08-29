import json
import pytest
from django.urls import reverse
from rest_framework.test import APIClient

@pytest.mark.django_db
def test_api_rank_endpoint(monkeypatch):
    from ml import apis

    # giả lập rank_cv_for_jd trả kết quả tĩnh
    def fake_rank_cv_for_jd(**kwargs):
        return [
            {"jd_id":"jd-abc","cv_id":"cv-1","pred":0.9,"score":0.9},
            {"jd_id":"jd-abc","cv_id":"cv-2","pred":0.1,"score":0.1},
        ]
    monkeypatch.setattr(apis, "rank_cv_for_jd", fake_rank_cv_for_jd)

    client = APIClient()
    payload = {
        "job_requirement": "Python, SQL",
        "job_description": "APIs",
        "candidates": [
            {"cv_id":"cv-1","resume_text":"Python dev"},
            {"cv_id":"cv-2","resume_text":"Java dev"}
        ],
        "topk": 2
    }
    resp = client.post("/api/rank/", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data and len(data["results"]) == 2
    assert data["results"][0]["cv_id"] == "cv-1"
