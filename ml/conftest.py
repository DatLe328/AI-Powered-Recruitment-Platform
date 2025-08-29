import pytest
import pandas as pd

@pytest.fixture
def tiny_pairs():
    jd = "Must-have: Python, FastAPI, PostgreSQL, Docker. Nice: AWS."
    cvs = [
        ("cv-1", "Python FastAPI PostgreSQL Docker; CI/CD; AWS ECS"),
        ("cv-2", "Java Spring Boot; MySQL; a bit of Docker"),
        ("cv-3", "Data engineer: Python, Kafka, PostgreSQL; Docker; K8s"),
        ("cv-4", "Frontend React Next.js TypeScript; Node.js"),
        ("cv-5", "DevOps: Docker Kubernetes; GitHub Actions; some Python AWS"),
    ]
    rows = []
    for cv_id, cv_text in cvs:
        rows.append({
            "jd_id": "demo-001",
            "job_description_text": jd,
            "cv_id": cv_id,
            "resume_text": cv_text,
        })
    return pd.DataFrame(rows)
