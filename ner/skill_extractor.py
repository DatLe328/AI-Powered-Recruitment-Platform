import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from sklearn.metrics import precision_score, recall_score, f1_score

# -------------------------- Utilities --------------------------
def is_upper_acronym(txt: str) -> bool:
    letters = [c for c in txt if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)

KNOWN_SHORT_SKILLS = {"C", "R", "C#", "C++", "SQL", "NLP", "AI", "ML", "DL", ".NET"}
STOPWORD_SINGLE_TOKEN = {
    "a","an","the","and","or","as","at","in","on","to","for","of","by","is","are","be","with","from"
}

def normalize_ws(text: str) -> str:
    return " ".join(str(text).split())

@dataclass
class Skill:
    skill_id: str
    name: str
    type: str
    length: int
    high_full: str
    high_abv: str
    low_forms: List[str]
    match_on_tokens: bool

# -------------------------- Indexer ----------------------------

class SkillIndex:
    def __init__(self, nlp, attr="LOWER"):
        self.nlp = nlp
        self.attr = attr
        # High/Low theo LOWER (không phân biệt hoa-thường)
        self.high = PhraseMatcher(nlp.vocab, attr=attr)
        self.low  = PhraseMatcher(nlp.vocab, attr=attr)
        # LOW acronym (từ low_forms nhìn như acronym) – case sensitive
        self.low_acronym = PhraseMatcher(nlp.vocab, attr="ORTH")
        # HIGH acronym (abbr “chính chủ” của skill) – case sensitive
        self.high_acronym = PhraseMatcher(nlp.vocab, attr="ORTH")

        self.skills: Dict[str, Skill] = {}
        # Reverse maps for debug/điều tra xung đột
        self._form_to_ids_high: Dict[str, List[str]] = {}
        self._form_to_ids_low: Dict[str, List[str]] = {}
        self._abv_vocab = set()

    def add_skill(self, skill: Skill):
        self.skills[skill.skill_id] = skill

        # ---- HIGH (full) theo LOWER
        patterns_high = []
        if skill.high_full:
            hf = normalize_ws(skill.high_full)
            if hf:
                patterns_high.append(hf)
                self._form_to_ids_high.setdefault(hf.lower(), []).append(skill.skill_id)

        # ---- HIGH acronym (ORTH) – KHÔNG nhét vào self.high để tránh "to" -> "TO"
        if skill.high_abv:
            abv = normalize_ws(skill.high_abv)
            if abv:
                self._abv_vocab.add(abv.lower())
                if is_upper_acronym(abv) or abv in KNOWN_SHORT_SKILLS:
                    # case-sensitive acronym: chỉ khớp đúng "SQL", "TO", "NLP", ...
                    self.high_acronym.add(skill.skill_id, [self.nlp.make_doc(abv)])
                else:
                    # fallback hiếm: viết tắt không phải acronym (để LOWER)
                    patterns_high.append(abv)
                    self._form_to_ids_high.setdefault(abv.lower(), []).append(skill.skill_id)

        if patterns_high:
            self.high.add(skill.skill_id, [self.nlp.make_doc(p) for p in patterns_high])

        # ---- LOW / LOW ACRONYM
        patterns_low_normal = []
        patterns_low_acr = []
        for lf in (skill.low_forms or []):
            lf_norm = normalize_ws(lf)
            if not lf_norm:
                continue
            if is_upper_acronym(lf_norm) or lf_norm in KNOWN_SHORT_SKILLS:
                # Để tránh nhiễu từ các acronym ngắn phổ biến, chỉ nhận nếu dài > 2 hoặc là known short
                if len(lf_norm) > 2 or lf_norm in KNOWN_SHORT_SKILLS:
                    patterns_low_acr.append(lf_norm)
            else:
                patterns_low_normal.append(lf_norm)

        if patterns_low_normal:
            self.low.add(skill.skill_id, [self.nlp.make_doc(p) for p in patterns_low_normal])
        if patterns_low_acr:
            self.low_acronym.add(skill.skill_id, [self.nlp.make_doc(p) for p in patterns_low_acr])

    @classmethod
    def from_skill_db(cls, nlp, skill_db: Dict[str, Any]):
        idx = cls(nlp, attr="LOWER")
        for sid, rec in skill_db.items():
            # high_surface_forms (hỗ trợ cả key sai chính tả)
            high = rec.get("high_surface_forms") or rec.get("high_surfce_forms") or {}
            if isinstance(high, dict):
                high_full = normalize_ws(high.get("full",""))
                high_abv  = normalize_ws(high.get("abv",""))
            elif isinstance(high, list):
                # đoán đơn giản: cái nào không phải acronym là full, acronym thì abv
                high_full, high_abv = "", ""
                for f in high:
                    f = normalize_ws(f)
                    if not f: 
                        continue
                    if not is_upper_acronym(f):
                        if not high_full: high_full = f
                    else:
                        if not high_abv:  high_abv  = f
            elif isinstance(high, str):
                high_full = normalize_ws(high)
                high_abv  = ""
            else:
                high_full = ""
                high_abv  = ""

            # Fallback: nếu chưa có high_full → lấy từ tên skill
            if not high_full:
                high_full = normalize_ws(rec.get("skill_name","") or rec.get("name",""))

            # NEW: nạp abbreviations từ DB cho “chính chủ”
            # Nếu DB không điền high_abv, mà có 'abbreviations', lấy cái hợp lệ đầu tiên
            if not high_abv:
                abrs = rec.get("abbreviations") or []
                if isinstance(abrs, str):
                    abrs = [abrs]
                for ab in abrs:
                    ab = normalize_ws(ab)
                    if ab and is_upper_acronym(ab):
                        high_abv = ab
                        break

            # low_forms như cũ
            low_raw = rec.get("low_surface_forms") or []
            if isinstance(low_raw, dict):
                low_forms = [normalize_ws(x) for x in low_raw.keys()]
            elif isinstance(low_raw, list):
                low_forms = [normalize_ws(x) for x in low_raw if x]
            elif isinstance(low_raw, str):
                low_forms = [normalize_ws(low_raw)] if low_raw else []
            else:
                low_forms = []

            skill = Skill(
                skill_id=sid,
                name=rec.get("skill_name","") or rec.get("name",""),
                type=rec.get("skill_type","Hard Skill"),
                length=int(rec.get("skill_len", 1)),
                high_full=high_full,
                high_abv=high_abv,       # <- giờ có thể đến từ 'abbreviations'
                low_forms=low_forms,
                match_on_tokens=bool(rec.get("match_on_tokens", False))
            )
            if (skill.high_full or skill.high_abv or skill.low_forms):
                idx.add_skill(skill)
        return idx

# -------------------------- Extractor --------------------------

def _to_span(doc: Doc, start: int, end: int) -> Tuple[int,int,str]:
    span = doc[start:end]
    return (span.start_char, span.end_char, span.text)

def _pick_best_spans(spans: List[Tuple[int,int,dict]], allow_overlaps=False) -> List[Tuple[int,int,dict]]:
    """
    Resolve overlaps. Priority (SkillNER-ish):
      1) source: high > acronym > low > token
      2) longer span wins (tie-break)
    """
    if allow_overlaps:
        return spans

    SOURCE_RANK = {"high": 3, "acronym": 2, "low": 1, "token": 0}

    def score(item):
        s,e,data = item
        length = e - s
        src = data.get("source","low")
        src_rank = SOURCE_RANK.get(src, 0)
        # Ưu tiên nguồn trước, sau đó mới xét độ dài span
        return (src_rank, length)

    spans_sorted = sorted(spans, key=score, reverse=True)
    chosen = []
    for s,e,d in spans_sorted:
        overlap = False
        for cs,ce,_ in chosen:
            if not (e <= cs or s >= ce):
                overlap = True
                break
        if not overlap:
            chosen.append((s,e,d))
    return sorted(chosen, key=lambda x: x[0])

class SkillExtractorFS:
    def __init__(self, enable_token_scanner=True, relax=0.2, allow_overlaps=False):
        # Có thể đổi sang blank('en') để nhẹ hơn:
        # self.nlp = spacy.blank("en"); self.nlp.add_pipe("sentencizer")
        self.nlp = spacy.load("en_core_web_lg", exclude=["ner"])
        self.relax = float(relax)
        self.allow_overlaps = allow_overlaps
        self.enable_token_scanner = enable_token_scanner
        with open('data/processed/skill_db.json', "r", encoding="utf-8") as f:
            skill_db = json.load(f)
        self.index = SkillIndex.from_skill_db(self.nlp, skill_db)

        # precompute tokenized forms for token-scanner
        self._token_skills = []
        for s in self.index.skills.values():
            if s.match_on_tokens and s.length > 2:
                toks = [t for t in s.high_full.lower().split() if t] if s.high_full else s.name.lower().split()
                # fallback: lấy low_forms[0] nếu không có full
                if not toks and s.low_forms:
                    toks = s.low_forms[0].lower().split()
                toks = [t for t in toks if t.isalnum() or len(t)>1]
                if toks:
                    self._token_skills.append((s.skill_id, toks))

    def annotate(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)

        # High (full) – LOWER
        high_hits = []
        for sid, starts_ends in self._match_phrase(self.index.high, doc):
            for start, end in starts_ends:
                s_char, e_char, span_text = _to_span(doc, start, end)
                high_hits.append((s_char, e_char, {
                    "skill_id": sid,
                    "source": "high",
                    "match": span_text
                }))

        # High acronym – ORTH (case-sensitive) → để SQL/TO/NLP thắng low/acr khác
        hi_acr_hits = []
        for sid, starts_ends in self._match_phrase(self.index.high_acronym, doc):
            for start, end in starts_ends:
                s_char, e_char, span_text = _to_span(doc, start, end)
                # nguồn vẫn là 'high' để đi vào full_matches
                hi_acr_hits.append((s_char, e_char, {
                    "skill_id": sid,
                    "source": "high",
                    "match": span_text
                }))

        # Low (text) – LOWER (lọc stopword cho 1-token chữ thường)
        low_hits = []
        for sid, starts_ends in self._match_phrase(self.index.low, doc):
            for start, end in starts_ends:
                s_char, e_char, span_text = _to_span(doc, start, end)
                if (end - start) == 1:
                    tok = span_text
                    if tok.islower() and tok.lower() in STOPWORD_SINGLE_TOKEN:
                        continue
                low_hits.append((s_char, e_char, {
                    "skill_id": sid,
                    "source": "low",
                    "match": span_text
                }))

        # Low acronym – ORTH (case-sensitive) → gán source='acronym' để ưu tiên dưới high
        acr_hits = []
        for sid, starts_ends in self._match_phrase(self.index.low_acronym, doc):
            for start, end in starts_ends:
                s_char, e_char, span_text = _to_span(doc, start, end)
                acr_hits.append((s_char, e_char, {
                    "skill_id": sid,
                    "source": "acronym",
                    "match": span_text
                }))

        # Token-scanner (optional)
        token_hits = []
        if self.enable_token_scanner and self._token_skills:
            token_hits = self._token_window_scanner(doc)

        # Ưu tiên gộp theo: high_acronym -> high -> acronym(low) -> low -> token
        all_hits = hi_acr_hits + high_hits + acr_hits + low_hits + token_hits
        final_spans = _pick_best_spans(all_hits, allow_overlaps=self.allow_overlaps)

        # Build outputs
        results = {
            "full_matches": [],
            "partial_matches": [],
            "token_scanner": []
        }
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

    # -------- helpers --------
    def _match_phrase(self, matcher: PhraseMatcher, doc: Doc) -> List[Tuple[str, List[Tuple[int,int]]]]:
        out = {}
        for m_id, start, end in matcher(doc):
            label = self.nlp.vocab.strings[m_id]
            out.setdefault(label, []).append((start, end))
        return list(out.items())

    def _token_window_scanner(self, doc: Doc) -> List[Tuple[int,int,dict]]:
        # Bag-of-words trong cửa sổ len(tokens) + floor(relax * len(tokens)).
        tokens = [t for t in doc if (t.is_alpha or t.is_digit) and not t.is_space]
        lower = [t.text.lower() for t in tokens]
        hits = []

        for sid, skill_toks in self._token_skills:
            L = len(skill_toks)
            if L < 3:
                continue
            min_needed = math.ceil((1.0 - self.relax) * L)
            win = L + max(0, int(self.relax * L))

            # sliding windows
            for i in range(0, max(0, len(lower)-1)):
                j = min(len(lower), i+win)
                window = lower[i:j]
                present = sum(1 for t in set(skill_toks) if t in window)
                if present >= min_needed:
                    s_char = tokens[i].idx
                    e_char = tokens[j-1].idx + len(tokens[j-1])
                    span_text = doc.text[s_char:e_char]
                    hits.append((s_char, e_char, {
                        "skill_id": sid,
                        "source": "token",
                        "match": span_text
                    }))
        return hits

    def get_training_data_from_annotation(self, annotation: Dict[str, Any]) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
        """
        Chuyển kết quả annotate thành dữ liệu train cho spaCy.
        """
        entities = []
        for group in ["full_matches", "partial_matches", "token_scanner"]:
            for item in annotation["results"][group]:
                entities.append((item["start"], item["end"], item["skill_type"]))
                # entities.append((item["start"], item["end"], "SKILL"))
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

def main():
    ap = argparse.ArgumentParser(description="Skill NER (improved matching/priority)")
    ap.add_argument("--text", help="Single input text")
    ap.add_argument("--file", help="Path to a UTF-8 text file to process")
    ap.add_argument("--relax", type=float, default=0.5, help="Relax ratio for token scanner (default 0.2)")
    ap.add_argument("--allow_overlaps", action="store_true", help="Keep overlapping spans (off by default)")
    ap.add_argument("--disable_token_scanner", action="store_true", help="Disable token-scanner step")
    ap.add_argument("--json_out", help="If set, write JSON result to this file")
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("Provide --text or --file")

    extractor = SkillExtractorFS(enable_token_scanner=not args.disable_token_scanner,
                                 relax=args.relax,
                                 allow_overlaps=args.allow_overlaps)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    out = extractor.annotate(text)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    validate()
    # main()
