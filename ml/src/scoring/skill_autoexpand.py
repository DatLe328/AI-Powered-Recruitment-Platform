import json, re, time, unicodedata, os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ===== Helper =====
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s)
                   if unicodedata.category(ch) != "Mn")
def nd(s: str) -> str:
    return strip_accents(s.lower())

_word = re.compile(r"[a-zA-ZÀ-ỹ0-9+#.+\-]{2,}", re.UNICODE)

VI_DIACRITIC_SET = set("ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
def looks_vietnamese(text: str) -> bool:
    t = text.lower()
    return any(ch in VI_DIACRITIC_SET for ch in t)

class SkillAutoExpander:
    def __init__(self,
                 skills_path: str = "skills_bilingual.json",
                 model_name: str = "intfloat/multilingual-e5-base",
                 sim_threshold: float = 0.90,
                 min_freq: int = 3,
                 blacklist: List[str] = None,
                 allow_autowrite: bool = True,
                 changelog_path: str = "skills_autoupdate.log.jsonl"):

        self.skills_path = skills_path
        self.model_name = model_name
        self.sim_threshold = sim_threshold
        self.min_freq = min_freq
        self.blacklist = set((blacklist or []) + ["c", "r", "go", "net", "ai", "ml"]) 
        self.allow_autowrite = allow_autowrite
        self.changelog_path = changelog_path

        with open(self.skills_path, "r", encoding="utf-8") as f:
            self.skills = json.load(f)

        self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        self.skill_id_to_text = {}
        self.skill_id_to_vec = {}
        if self.model:
            texts = []
            ids = []
            for sk in self.skills:
                sid = sk["skill_id"]
                t = f"{sk.get('name_en','')} || {sk.get('name_vi','')}"
                self.skill_id_to_text[sid] = t.strip()
                ids.append(sid); texts.append(t.strip())
            import numpy as np
            self.skill_mat = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.skill_ids = ids
        else:
            self.skill_mat, self.skill_ids = None, []

        self.alias_all = set()
        for sk in self.skills:
            for k in ("aliases_en", "aliases_vi", "aliases_nd"):
                for a in sk.get(k, []):
                    self.alias_all.add(a.lower())

    def mine_candidates(self, text: str, ngram=(1,2,3,4)) -> Counter:
        toks = _word.findall(text.lower())
        cands = Counter()
        for n in range(ngram[0], ngram[1]+1):
            for i in range(len(toks)-n+1):
                phrase = " ".join(toks[i:i+n]).strip()
                if len(phrase) < 2: continue
                if phrase in self.blacklist: continue
                if phrase in self.alias_all: continue
                # bỏ pure numeric
                if re.fullmatch(r"[0-9.]+", phrase): continue
                cands[phrase] += 1
        return cands

    def _nearest_skill(self, phrase: str) -> Tuple[str, float]:
        if not self.model or self.skill_mat is None:
            return "", 0.0
        import numpy as np
        v = self.model.encode([phrase], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = (self.skill_mat @ v)
        j = int(np.argmax(sims))
        return self.skill_ids[j], float(sims[j])

    def _append_alias(self, sid: str, alias: str):
        is_vi = looks_vietnamese(alias)
        for sk in self.skills:
            if sk["skill_id"] == sid:
                if is_vi:
                    sk.setdefault("aliases_vi", [])
                    sk.setdefault("aliases_nd", [])
                    if alias.lower() not in [a.lower() for a in sk["aliases_vi"]]:
                        sk["aliases_vi"].append(alias)
                    nd_alias = nd(alias)
                    if nd_alias not in [a.lower() for a in sk["aliases_nd"]]:
                        sk["aliases_nd"].append(nd_alias)
                else:
                    sk.setdefault("aliases_en", [])
                    if alias.lower() not in [a.lower() for a in sk["aliases_en"]]:
                        sk["aliases_en"].append(alias)
                break

    def _save_dict(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{os.path.splitext(self.skills_path)[0]}.bak.{ts}.json"
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)
        with open(self.skills_path, "w", encoding="utf-8") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)

    def _log_change(self, record: Dict):
        with open(self.changelog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def suggest_and_update(self, texts: List[str]) -> Dict:
        pool = Counter()
        for t in texts:
            pool.update(self.mine_candidates(t, ngram=(1,4)))

        accepted, rejected = [], []
        for phrase, freq in pool.items():
            if freq < self.min_freq:
                continue
            sid, sim = self._nearest_skill(phrase)
            if len(phrase) <= 2: continue
            if phrase in self.blacklist: continue
            if sim >= self.sim_threshold:
                rec = {"ts": int(time.time()), "alias": phrase, "skill_id": sid,
                       "similarity": round(sim, 4), "freq": freq, "action": "accepted" if self.allow_autowrite else "proposed"}
                accepted.append(rec)
                self._log_change(rec)
                if self.allow_autowrite:
                    self._append_alias(sid, phrase)
            else:
                rejected.append({"alias": phrase, "skill_id": sid, "similarity": round(sim,4), "freq": freq})

        if self.allow_autowrite and accepted:
            self._save_dict()

        return {"accepted": accepted, "rejected": rejected, "threshold": self.sim_threshold, "min_freq": self.min_freq}