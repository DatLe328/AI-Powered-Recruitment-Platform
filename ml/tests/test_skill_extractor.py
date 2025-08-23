import pytest
import ml.src.scoring as se_mod
from collections import Counter

@pytest.fixture
def skill_dict():
    return {
        "python": {"aliases": ["py", "python3"]},
        "fastapi": {"aliases": ["fast api", "fast-api"]},
        "postgresql": {"aliases": ["postgres", "postgre sql", "postgres sql"]},
        "docker": {"aliases": []},
        "aws": {"aliases": ["amazon web services"]},
    }

def _make_extractor(skill_dict):
    SkillExtractor = getattr(se_mod, "SkillExtractor")
    try:
        return SkillExtractor(skill_dict=skill_dict)
    except TypeError:
        try:
            return SkillExtractor(vendor_skill_dict=skill_dict)
        except Exception:
            inst = SkillExtractor()
            if hasattr(inst, "set_skill_dict"):
                inst.set_skill_dict(skill_dict)
            elif hasattr(inst, "skill_dict"):
                inst.skill_dict = skill_dict
            return inst

def _to_iter(xs):
    if xs is None:
        return []
    if isinstance(xs, (list, set, tuple)):
        return list(xs)
    return []

def test_extract_basic(skill_dict):
    extr = _make_extractor(skill_dict)
    text = "Experienced in PYTHON3 and Fast API; using Postgres + Docker on Amazon Web Services."
    skills = set(_to_iter(extr.extract(text)))
    assert {"python","fastapi","postgresql","docker","aws"} <= skills
    assert skills.isdisjoint({"java","spring","react"})

def test_extract_alias_dedup(skill_dict):
    extr = _make_extractor(skill_dict)
    text = "py python Python3; fast-api FAST API; postgres postgre sql"
    skills = _to_iter(extr.extract(text))
    cnt = Counter(skills)
    for k in ["python","fastapi","postgresql"]:
        assert cnt[k] == 1

def test_empty_text(skill_dict):
    extr = _make_extractor(skill_dict)
    out = _to_iter(extr.extract(""))
    assert len(out) == 0