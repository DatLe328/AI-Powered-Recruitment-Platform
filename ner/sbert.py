# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

# ---------- Optional: SBERT ----------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # keep import optional
    SentenceTransformer = None
    np = None


# ============================= Utils =============================

_NORM_KEEP = r"[^\w\s\+\#\.\-]"  # keep + # . - for C++, C#, .NET etc.


def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s).lower()
    # drop things in parentheses/brackets
    s = re.sub(r"\s*[\(\[\{][^)\]\}]*[\)\]\}]", "", s)
    # remove unwanted punct
    s = re.sub(_NORM_KEEP, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_safe_abbr(x: str) -> bool:
    return bool(x) and x.isupper() and len(x) >= 3


# ============================= Data =============================

@dataclass
class SkillDef:
    skill_id: str
    name: str
    type: str = "Specialized Skill"            # "Certification" or other
    high_full: str = ""                 # canonical full phrase
    low_forms: List[str] = field(default_factory=list)  # other aliases/lemmas/stems
    match_on_tokens: bool = False       # enable token-window scanning
    length: int = 0                     # token length of high_full


class SkillIndex:
    """
    Index container: vocab, phrase patterns, acronym map, token-forms
    """
    def __init__(self, nlp: "spacy.language.Language"):
        self.nlp = nlp
        self.skills: Dict[str, SkillDef] = {}
        self.phrase = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.phrase_labels: Dict[str, str] = {}  # surface text -> LABEL (SKILL/CERTIFICATION)
        self.acronym_map: Dict[str, List[str]] = {}  # abbr(lower) -> [skill_id]
        self._added = False

    @staticmethod
    def _label_of(t: str) -> str:
        return "CERTIFICATION" if (isinstance(t, str) and t.strip().lower() == "certification") else "SKILL"

    @classmethod
    def from_skill_db(cls, nlp, db: Dict) -> "SkillIndex":
        """
        Accepts flexible skill DBs:
          - light-style: {id: {skill_name/skill_cleaned/skill_type/skill_lemmed/skill_stemmed/...}}
          - or your custom 'skill_db.json' with {id: {name, type, high_full, low_forms, abbreviation? ...}}
        """
        inst = cls(nlp)
        surfaces_by_label: Dict[str, List[str]] = {"SKILL": [], "CERTIFICATION": []}

        for sid, item in db.items():
            # robust field mapping
            name = item.get("name") or item.get("skill_name") or ""
            stype = item.get("type") or item.get("skill_type") or ""
            label = inst._label_of(stype)

            # collect surface forms
            forms = []
            # prefer explicit high_full then derive from known fields
            high_full = item.get("high_full") or item.get("skill_cleaned") or item.get("skill_name") or ""
            if high_full:
                forms.append(high_full)
            # add lemmas/stems/low_forms
            for k in ("low_forms", "skill_lemmed", "skill_stemmed"):
                v = item.get(k)
                if not v:
                    continue
                if isinstance(v, list):
                    forms.extend(v)
                else:
                    forms.append(v)

            # normalize + dedupe
            forms_norm = []
            seen = set()
            for s in forms:
                s = normalize_text(s)
                if s and s not in seen:
                    seen.add(s)
                    forms_norm.append(s)

            # build skill def
            high_norm = forms_norm[0] if forms_norm else normalize_text(name)
            toks = [t.text for t in nlp.make_doc(high_norm)]
            sd = SkillDef(
                skill_id=str(sid),
                name=name or high_norm,
                type=stype or ("Certification" if label == "CERTIFICATION" else "Hard Skill"),
                high_full=high_norm,
                low_forms=forms_norm[1:],
                match_on_tokens=True if len(toks) > 2 else False,
                length=len(toks),
            )
            inst.skills[sd.skill_id] = sd

            # phrase label mapping
            surfaces_by_label[label].append(high_norm)
            for lf in sd.low_forms[:2]:  # keep index small (top-2)
                surfaces_by_label[label].append(lf)

            # acronym (optional if DB has one)
            abbr = (item.get("abbreviation") or "").strip()
            if is_safe_abbr(abbr):
                inst.acronym_map.setdefault(abbr.lower(), []).append(sd.skill_id)

        # Add phrase patterns
        for lab, arr in surfaces_by_label.items():
            arr = list(dict.fromkeys(arr))
            if not arr:
                continue
            docs = list(nlp.pipe(arr))
            inst.phrase.add(lab, docs)
            for s in arr:
                inst.phrase_labels[s] = lab
        inst._added = True
        return inst


# ============================= Extractor =============================

def _source_rank(name: str) -> int:
    # Higher is better. Adjust to taste.
    table = {"high": 4, "acronym": 3, "semantic": 3, "low": 2, "token": 1}
    return table.get(name, 0)


def _pick_best_spans(hits: List[Tuple[int, int, dict]], allow_overlaps: bool = False
                     ) -> List[Tuple[int, int, dict]]:
    """
    hits: list of (start_char, end_char, data=dict(skill_id, source, match, score, ...))
    - prefer longer spans, then higher source rank, then higher score, then earlier position
    """
    if not hits:
        return []

    # sort by: start asc (stable), then end-start desc, then rank desc, then score desc
    def keyfn(h):
        s, e, d = h
        return (s, -(e - s), -_source_rank(d.get("source", "")), -float(d.get("score", 0.0)))

    hits_sorted = sorted(hits, key=keyfn)

    if allow_overlaps:
        return hits_sorted

    final = []
    taken = []
    for s, e, d in hits_sorted:
        overlapped = False
        for (ts, te, _) in taken:
            if not (e <= ts or s >= te):  # overlap
                overlapped = True
                break
        if not overlapped:
            final.append((s, e, d))
            taken.append((s, e, d))
    return final


class SkillExtractorFS:
    """
    Multi-tier extractor:
      - high (PhraseMatcher exact phrases from high_full + (subset) low_forms)
      - acronym (from DB abbreviation if present)
      - low (relaxed, token-window coverage)
      - token_scanner (looser token windows)
      - semantic (optional SBERT re-call)
    """

    def __init__(self,
                 enable_token_scanner: bool = True,
                 relax: float = 0.2,
                 allow_overlaps: bool = False,
                 model_name: str = "en_core_web_lg",
                 # SBERT options
                 use_sbert: bool = False,
                 sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 sbert_top_k: int = 3,
                 sbert_min_cosine: float = 0.72,
                 sbert_batch_size: int = 32,
                 # DB path options
                 skill_db_path: str = "data/processed/skill_db.json"):
        try:
            self.nlp = spacy.load(model_name, exclude=["ner"])
        except Exception:
            # Fall back to blank if model missing (keeps tokenizer)
            self.nlp = spacy.blank("en")

        self.relax = float(relax)
        self.allow_overlaps = allow_overlaps
        self.enable_token_scanner = enable_token_scanner

        # Load skill DB (robust to structure)
        with open(skill_db_path, "r", encoding="utf-8") as f:
            skill_db = json.load(f)
        self.index = SkillIndex.from_skill_db(self.nlp, skill_db)

        # Precompute token skills for low/token scanners
        self._token_skills: List[Tuple[str, List[str]]] = []
        for s in self.index.skills.values():
            toks = [t.text for t in self.nlp.make_doc(s.high_full)]
            if not toks and s.low_forms:
                toks = [t.text for t in self.nlp.make_doc(s.low_forms[0])]
            toks = [t for t in toks if t]  # remove empties
            if toks:
                self._token_skills.append((s.skill_id, toks))

        # ---------- SBERT semantic tier ----------
        self.use_sbert = bool(use_sbert)
        self.sbert_top_k = int(sbert_top_k)
        self.sbert_min_cosine = float(sbert_min_cosine)
        self.sbert_batch_size = int(sbert_batch_size)

        self._sbert = None
        self._skill_ids: List[str] = []
        self._skill_texts: List[str] = []
        self._skill_vecs = None  # (N, D) L2-normalized

        if self.use_sbert:
            if SentenceTransformer is None or np is None:
                raise RuntimeError("sentence-transformers / numpy not installed")
            self._sbert = SentenceTransformer(sbert_model)
            self._build_sbert_index(skill_db)

    # -------------------- SBERT Index --------------------

    def _build_sbert_index(self, skill_db: Dict):
        texts = []
        ids = []
        for sid, s in self.index.skills.items():
            reps = []
            if s.high_full:
                reps.append(s.high_full)
            for lf in s.low_forms[:2]:
                reps.append(lf)
            reps = [t for t in {normalize_text(x) for x in reps} if t]
            if not reps:
                continue
            # join reps to one text; or encode each rep then mean (slower but OK)
            merged = " ; ".join(reps)
            texts.append(merged)
            ids.append(sid)

        if not texts:
            return
        vecs = self._sbert.encode(texts, batch_size=self.sbert_batch_size,
                                  show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        self._skill_ids = ids
        self._skill_texts = texts
        self._skill_vecs = vecs

    def _iter_semantic_candidates(self, doc: Doc):
        # noun chunks (if parser available)
        if "parser" in self.nlp.pipe_names:
            for chunk in doc.noun_chunks:
                t = chunk.text.strip()
                if len(t) >= 3:
                    yield (chunk.start_char, chunk.end_char, t)
        # 2..5-grams over alpha/digit tokens
        toks = [t for t in doc if (t.is_alpha or t.is_digit)]
        for i in range(len(toks)):
            for n in (5, 4, 3, 2):
                if i + n <= len(toks):
                    s = toks[i].idx
                    e = toks[i + n - 1].idx + len(toks[i + n - 1].text)
                    txt = doc.text[s:e].strip()
                    if len(txt) >= 3:
                        yield (s, e, txt)

    def _sbert_candidates(self, doc: Doc):
        if not (self.use_sbert and self._skill_vecs is not None):
            return []
        spans = []
        buf_texts = []
        buf_meta = []
        seen = set()
        for s, e, txt in self._iter_semantic_candidates(doc):
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            buf_texts.append(txt)
            buf_meta.append((s, e, txt))

        if not buf_texts:
            return []

        qvecs = self._sbert.encode(buf_texts, batch_size=self.sbert_batch_size,
                                   show_progress_bar=False, convert_to_numpy=True)
        qnorm = np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-12
        qvecs = qvecs / qnorm
        sims = np.matmul(qvecs, self._skill_vecs.T)  # (Q, N)

        hits = []
        for i, (s, e, txt) in enumerate(buf_meta):
            row = sims[i]
            k = min(self.sbert_top_k, row.shape[0] - 1) if row.shape[0] > 1 else 1
            top_idx = np.argpartition(-row, k)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            for j in top_idx:
                score = float(row[j])
                if score < self.sbert_min_cosine:
                    continue
                sid = self._skill_ids[j]
                hits.append((s, e, {"skill_id": sid, "source": "semantic", "match": txt, "score": score}))
        return hits

    # -------------------- Lexical tiers --------------------

    def _high_phrase_hits(self, doc: Doc) -> List[Tuple[int, int, dict]]:
        out = []
        m = self.index.phrase(doc)
        for mid, s, e in m:
            label = self.nlp.vocab.strings[mid]  # "SKILL"/"CERTIFICATION"
            span = doc[s:e]
            txt_norm = normalize_text(span.text)
            out.append((span.start_char, span.end_char,
                        {"skill_id": self._find_skill_id_by_surface(txt_norm, label),
                         "source": "high", "match": span.text, "score": 1.0}))
        return out

    def _find_skill_id_by_surface(self, surface_norm: str, label: str) -> str:
        # naive reverse lookup (fast enough for downstream; can index by surface->id for speed)
        for sid, s in self.index.skills.items():
            if label == "CERTIFICATION" and s.type.lower() != "certification":
                continue
            if surface_norm == s.high_full:
                return sid
            if surface_norm in s.low_forms:
                return sid
        return ""

    def _acronym_hits(self, doc: Doc) -> List[Tuple[int, int, dict]]:
        out = []
        if not self.index.acronym_map:
            return out
        for t in doc:
            tok = t.text.strip()
            if is_safe_abbr(tok):
                sid_list = self.index.acronym_map.get(tok.lower()) or []
                for sid in sid_list:
                    out.append((t.idx, t.idx + len(tok),
                                {"skill_id": sid, "source": "acronym", "match": tok, "score": 0.96}))
        return out

    def _low_relax_hits(self, doc):
        """
        Relaxed token-window: subsequence theo đúng thứ tự, nhưng
        - với mẫu 2 token: phải trúng cả 2 (không cho rơi)
        - với mẫu >2 token: cho rơi <= k và yêu cầu >=2 token trúng
        Span lấy theo token match đầu tiên -> token match cuối cùng (không lấy cả cửa sổ).
        """
        import math
        out = []
        doc_toks = [normalize_text(t.text) for t in doc]

        for sid, s_toks0 in self._token_skills:
            s_toks = [normalize_text(x) for x in s_toks0 if normalize_text(x)]
            L = len(s_toks)
            if L <= 1:
                continue

            k = int(math.ceil(L * self.relax))
            for i in range(len(doc_toks)):
                # seed theo token đầu tiên
                if doc_toks[i] != s_toks[0]:
                    continue

                hits = 0
                first_hit = None
                last_hit = None
                j = i
                p = 0
                # quét tối đa L+k token tính từ i
                while j < len(doc_toks) and p < L and (j - i) <= (L + k):
                    if doc_toks[j] == s_toks[p]:
                        if first_hit is None:
                            first_hit = j
                        last_hit = j
                        hits += 1
                        p += 1
                    j += 1

                # điều kiện đủ
                enough = (L == 2 and hits == 2) or (L > 2 and hits >= max(L - k, 2))
                if not enough or first_hit is None or last_hit is None:
                    continue

                # span = token match đầu tiên -> token match cuối cùng
                start_char = doc[first_hit].idx
                end_char = doc[last_hit].idx + len(doc[last_hit].text)
                sp = doc.char_span(start_char, end_char, alignment_mode="contract")
                if sp is None:
                    sp = doc.char_span(start_char, end_char, alignment_mode="expand")
                if sp is not None:
                    score = 0.6 + 0.4 * (hits / L)
                    out.append((sp.start_char, sp.end_char, {
                        "skill_id": sid, "source": "low", "match": sp.text, "score": score
                    }))
        return out

    def _token_scanner_hits(self, doc):
        """
        Scanner lỏng hơn: yêu cầu TẤT CẢ token của mẫu xuất hiện trong cửa sổ (bất kỳ thứ tự),
        và span là **đoạn phủ nhỏ nhất** của các token trúng (không dùng cả cửa sổ).
        """
        if not self.enable_token_scanner:
            return []
        import math
        out = []
        doc_toks = [normalize_text(t.text) for t in doc]

        for sid, s_toks0 in self._token_skills:
            s_list = [normalize_text(x) for x in s_toks0 if normalize_text(x)]
            # bỏ trùng & tính kích thước mẫu
            seen = set()
            s_seq = [t for t in s_list if not (t in seen or seen.add(t))]
            L = len(s_seq)
            if L <= 2:
                continue

            k = int(math.ceil(L * self.relax))
            win = L + k

            for i in range(len(doc_toks)):
                W = doc_toks[i:i + win]
                if len(W) < L:
                    break

                # map token -> các vị trí trong cửa sổ
                pos = {}
                for off, tok in enumerate(W):
                    if tok in s_seq:
                        pos.setdefault(tok, []).append(i + off)

                if not all(tok in pos for tok in s_seq):
                    continue

                # chọn vị trí sớm nhất cho mỗi token -> đoạn phủ nhỏ nhất
                idxs = [pos[tok][0] for tok in s_seq]
                start_idx = min(idxs)
                end_idx = max(idxs)

                start_char = doc[start_idx].idx
                end_char = doc[end_idx].idx + len(doc[end_idx].text)
                sp = doc.char_span(start_char, end_char, alignment_mode="contract")
                if sp is None:
                    sp = doc.char_span(start_char, end_char, alignment_mode="expand")
                if sp is not None:
                    out.append((sp.start_char, sp.end_char, {
                        "skill_id": sid, "source": "token", "match": sp.text, "score": 0.5
                    }))
        return out

    # -------------------- Public API --------------------

    def annotate(self, text: str) -> Dict:
        doc = self.nlp.make_doc(text)

        high_hits = self._high_phrase_hits(doc)
        acr_hits = self._acronym_hits(doc)
        low_hits = self._low_relax_hits(doc)
        token_hits = self._token_scanner_hits(doc)

        all_hits = high_hits + acr_hits + low_hits + token_hits

        # SBERT tier
        semantic_hits = []
        if self.use_sbert:
            semantic_hits = self._sbert_candidates(doc)
            all_hits += semantic_hits

        final_spans = _pick_best_spans(all_hits, allow_overlaps=self.allow_overlaps)

        results = {"full_matches": [], "partial_matches": [], "token_scanner": []}
        seen = set()
        for s_char, e_char, data in final_spans:
            sid = data.get("skill_id", "")
            if not sid:
                continue
            sdef = self.index.skills.get(sid)
            if not sdef:
                continue

            item = {
                "skill_id": sid,
                "skill_name": sdef.name,
                "skill_type": sdef.type,
                "start": s_char,
                "end": e_char,
                "match": data.get("match", ""),
                "source": data.get("source", "low"),
                "score": float(data.get("score", 0.0)),
            }

            # route to buckets (keep semantic with partial to stay backward compatible)
            if item["source"] == "high":
                results["full_matches"].append(item)
            elif item["source"] == "token":
                results["token_scanner"].append(item)
            else:
                results["partial_matches"].append(item)

            seen.add((s_char, e_char, sid))

        unique_skills = {}

        for s_char, e_char, data in final_spans:
            sid = data["skill_id"]
            sdef = self.index.skills.get(sid)
            if not sdef:
                continue
            item = {
                "skill_id": sid,
                "skill_name": sdef.name,
                "skill_type": sdef.type,
                "start": s_char,
                "end": e_char,
                "match": data.get("match",""),
                "source": data.get("source","low")
            }
            if item["source"] == "high":
                results["full_matches"].append(item)
            elif item["source"] == "token":
                results["token_scanner"].append(item)
            else:
                results["partial_matches"].append(item)

            uni = unique_skills.setdefault(sid, {
                "skill_name": sdef.name,
                "skill_type": sdef.type,
                "occurrences": 0
            })
            uni["occurrences"] += 1
        return {
            "text": text,
            "results": results,
            "unique_skills": unique_skills
        }

    def get_training_data_from_annotation(
        self,
        annotation: Dict[str, Any]
    ) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
        import json

        # 1) Chấp nhận annotation dạng chuỗi JSON
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except Exception:
                return ("", {"entities": []})

        # 2) Lấy text & results an toàn
        text = ""
        results = {}
        if isinstance(annotation, dict):
            text = str(annotation.get("text", "") or "")
            res = annotation.get("results", {})
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except Exception:
                    res = {}
            if isinstance(res, dict):
                results = res

        # 3) Duyệt 3 nhóm như code gốc
        raw_items: List[Tuple[int, int, str, str]] = []  # (start,end,label,match_text)
        for group in ("full_matches", "partial_matches", "token_scanner"):
            items = results.get(group, [])
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except Exception:
                    items = []
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    # Cho phép dạng [s,e,label]
                    if isinstance(it, (list, tuple)) and len(it) == 3:
                        try:
                            s, e = int(it[0]), int(it[1])
                            lb = str(it[2])
                            raw_items.append((s, e, lb, ""))
                        except Exception:
                            pass
                    continue
                s, e = it.get("start"), it.get("end")
                try:
                    s, e = int(s), int(e)
                except Exception:
                    continue
                lb = str(it.get("skill_type") or it.get("label") or "SKILL")
                mtxt = str(it.get("match") or "")
                raw_items.append((s, e, lb, mtxt))

        # 4) Snap/re-anchor về đúng biên token trên văn bản gốc
        from spacy.tokens import Span
        doc = self.nlp.make_doc(text)
        spans: List[Span] = []

        for s, e, lb, mtxt in raw_items:
            sp = doc.char_span(s, e, label=lb, alignment_mode="contract")
            if sp is None:
                # thử expand nhẹ
                sp = doc.char_span(s, e, label=lb, alignment_mode="expand")
            if sp is None and mtxt:
                # 5) Re-anchor: tìm 'match' quanh vị trí dự kiến (± 64 ký tự)
                w0 = max(0, s - 64)
                w1 = min(len(text), e + 64 if e > s else s + 64)
                idx = text.find(mtxt, w0, w1)
                if idx != -1:
                    sp = doc.char_span(idx, idx + len(mtxt), label=lb, alignment_mode="contract")
                    if sp is None:
                        sp = doc.char_span(idx, idx + len(mtxt), label=lb, alignment_mode="expand")
            if sp is not None:
                spans.append(sp)

        # 6) Loại chồng lấn giống spaCy (ưu tiên span dài/ổn định)
        ents = filter_spans(spans)

        # 7) Xuất đúng format cũ
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in ents]

        return (annotation["text"], {"entities": entities})

def load_spacy_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_data = []
    for sample in data:
        text = sample["text"]
        true_skills = []
        for start, end, label in sample["entities"]:
            if label == "SKILL":
                true_skills.append(text[start:end])
        test_data.append({"text": text, "true_skills": true_skills})
    return test_data

def load_test_json(path):
    """
    Đọc file JSON test gồm danh sách dict {text: ..., skills: [...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # đảm bảo đúng format
    test_data = []
    for sample in data:
        text = sample.get("text", "")
        skills = sample.get("skills", [])
        test_data.append({"text": text, "skills": skills})
    return test_data

def evaluate_extractor(extractor, test_data):
    y_true_all = []
    y_pred_all = []

    # Lấy tất cả skill có trong ground truth + prediction để làm vocab
    all_skills = set()
    for sample in test_data:
        all_skills.update(sample["skills"])
        pred_result = extractor.annotate(sample["text"])
        pred_skills = [v["skill_name"] for v in pred_result["unique_skills"].values()]
        all_skills.update(pred_skills)
    all_skills = list(all_skills)

    # So sánh từng sample
    for sample in test_data:
        true_skills = set(sample["skills"])
        pred_result = extractor.annotate(sample["text"])
        print(pred_result)
        pred_skills = set(v["skill_name"] for v in pred_result["unique_skills"].values())

        # encode 0/1 vector cho toàn bộ vocab
        y_true = [1 if skill in true_skills else 0 for skill in all_skills]
        y_pred = [1 if skill in pred_skills else 0 for skill in all_skills]

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    # metrics
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}
# -------------------------- CLI -------------------------------

def validate():
    test_data = load_test_json("data/test/test_data.json")
    print(test_data)
    extractor = SkillExtractorFS(enable_token_scanner=True,
                                relax=0.5,
                                allow_overlaps=False)
    # Đánh giá
    metrics = evaluate_extractor(extractor, test_data)
    print(metrics)
# ============================= CLI (optional) =============================
def main():
    import argparse
    text = """
Developing and maintaining websites on servers running Microsoft SharePoint Server and Internet Information Services (IIS). Supporting Systems Management Server (SMS) Troubleshooting LAN, WAN, Internet, and Intranet network and security access. Troubleshooting network connectivity issues related to TCP/IP, Domain Name Service (DNS), Dynamic Host Configuration Protocol (DHCP) protocols, Internet Security and Acceleration (ISA) proxy server, and VPN. Troubleshooting web application/page issues, client browsers, and related software. Administering and maintaining of end user accounts, permissions, and access rights in in Microsoft Active Directory. Administering and maintaining of NTFS security permissions on the file servers. Installing, configuring, and maintaining hardware such as: servers, workstations, laptops, printers, and scanners in a Windows Enterprise environment. Installing, configuring, and supporting printers on the print servers. Installing, configuring, and supporting Microsoft Windows Server 2000 and 2003, Microsoft Windows XP and Windows Vista, and Microsoft Office XP, 2003, and 2007. Education Bachelor of Science , Information Technology 2005 Florida International Univeristy City , State , United States Coursework in Programming, Web Administration, Network Administration, Database Administration, and Systems Administration Linux Programming Languages: C++, Java, JSP, HTML, CSS, VB.Net, Bash, T-SQL Certifications CompTIA Network+ - 2014 Skills Active Directory, Azure, anti-virus, Backup Exec, backup, Bash, batch, Cacti, Cisco ASA, databases, DHCP, DNS, documentation, DataDomain, EMC, Enterprise Vault, ePO, file servers, firewall, GPO, HTML, IIS, ISA, LDAP, Linux, McAfee, Exchange, Microsoft Office, Microsoft Windows, security, policies, PowerShell, programming, proxy server, servers, scripts, SolarWinds, SQL, StorSimple, troubleshooting, TMG, Ubuntu, Visual Basic Script, VBS, Veritas Netbackup, VPN, VRanger, Veeam, VMWare, VDI, virtual manchine, NMap, ZenMap.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=text)
    ap.add_argument("--skill-db", default="data/processed/skill_db.json")
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument("--relax", type=float, default=0.2)
    ap.add_argument("--allow-overlaps", action="store_true")
    ap.add_argument("--token-scanner", action="store_true")
    ap.add_argument("--use-sbert", action="store_true")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-top-k", type=int, default=3)
    ap.add_argument("--sbert-min-cos", type=float, default=0.72)
    args = ap.parse_args()

    ext = SkillExtractorFS(
        enable_token_scanner=args.token_scanner,
        relax=args.relax,
        allow_overlaps=args.allow_overlaps,
        model_name=args.model,
        use_sbert=args.use_sbert,
        sbert_model=args.sbert_model,
        sbert_top_k=args.sbert_top_k,
        sbert_min_cosine=args.sbert_min_cos,
        skill_db_path=args.skill_db
    )
    out = ext.annotate(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    validate()