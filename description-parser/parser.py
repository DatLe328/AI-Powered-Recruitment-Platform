import re
import spacy
from spacy.pipeline import EntityRuler
from sentence_transformers import SentenceTransformer, util

# ========= Load vocab ==========
with open("data/vocab_base.txt", encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

with open("data/esco_raw_skills.txt", encoding="utf-8") as f:
    candidate_vocab = [line.strip() for line in f if line.strip()]
# ========= spaCy Exact Match ==========
def spacy_skill_parser(text, vocab):
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")

    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)

    patterns = [
        {"label": "SKILL", "pattern": [{"LOWER": token.lower()} for token in skill.split()]}
        for skill in vocab
    ]
    ruler.add_patterns(patterns)

    doc = nlp(text)

    print("\n🎯 Exact match (spaCy):")
    exact_skills = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            print(f"✅ {ent.text} [{ent.start_char}-{ent.end_char}]")
            exact_skills.add(ent.text.lower())
    return list(exact_skills), doc

# ========= Sentence-BERT ==========
def sentence_bert_skill_parser(doc, text, exact_skills, vocab):
    print("\n🎯 Semantic match (S-BERT):")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    skill_embeddings = model.encode(vocab, convert_to_tensor=True)

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        sentences = [line.strip() for line in text.split("\n") if line.strip()]

    skill_candidates = []
    for sentence in sentences:
        sentence_emb = model.encode(sentence, convert_to_tensor=True)
        cos_scores = util.cos_sim(sentence_emb, skill_embeddings)[0]
        top_idx = cos_scores.argmax().item()
        top_score = cos_scores[top_idx].item()
        skill_candidate = vocab[top_idx]

        if skill_candidate.lower() not in exact_skills and top_score > 0.4:
            print(f"🔷 {sentence}")
            print(f"    → Closest skill: {skill_candidate} (score: {top_score:.2f})")
            skill_candidates.append(skill_candidate.lower())
    return skill_candidates

# ========= Fit Score ==========
def _compute_fit_score(skills_cv, skills_jd):
    print("\n🎯 Tính Fit Score với Sentence-BERT:")
    print("📄 Skills từ CV:", skills_cv)
    print("📄 Skills từ JD:", skills_jd)

    # lowercase + loại trùng
    skills_cv = list(set([s.lower() for s in skills_cv]))
    skills_jd = list(set([s.lower() for s in skills_jd]))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings_cv = model.encode(skills_cv, convert_to_tensor=True)
    embeddings_jd = model.encode(skills_jd, convert_to_tensor=True)

    fit_scores = []
    detailed = []

    for i, jd_emb in enumerate(embeddings_jd):
        cos_scores = util.cos_sim(jd_emb, embeddings_cv)[0]
        best_idx = cos_scores.argmax().item()
        best_score = cos_scores[best_idx].item()
        matched_skill = skills_cv[best_idx]

        print(f"✅ JD Skill: {skills_jd[i]} → CV: {matched_skill} (score: {best_score:.2f})")
        fit_scores.append(best_score)
        detailed.append({
            "jd_skill": skills_jd[i],
            "cv_best_match": matched_skill,
            "similarity": best_score
        })

    final_score = (sum(fit_scores) / len(fit_scores)) * 100 if fit_scores else 0
    print(f"\n🎯 Fit Score (S-BERT): {final_score:.2f}%")

    return final_score, detailed

def compute_fit_score(cv, jd):
    skills_jd, doc = spacy_skill_parser(jd, vocab)
    skills_jd = skills_jd + sentence_bert_skill_parser(doc, jd, skills_jd, candidate_vocab)

    skills_cv, doc = spacy_skill_parser(cv, vocab)
    skills_cv = skills_cv + sentence_bert_skill_parser(doc, cv, skills_cv, candidate_vocab)

    return _compute_fit_score(skills_cv, skills_jd)

# ========= Main ==========
if __name__ == "__main__":
    text_cv = """
    - A STEM Degree or equivalent certification, in Computer Science, Information Systems, Fintech or Information Technology Or A degree in Auditing/ Accounting/ Business Management Studies with strong interest in data and technology

- Basic knowledge of popular systems such as ERP, accounting systems and how it supports the business process.

- Fluent language skills in English (both written and spoken) is mandatory.

- Proficient with at least one of the following tools Microsoft Excel, SQL, ACL or Alteryx.


And other perks at EY

- Continuous learning: You’ll develop the mindset and skills to navigate whatever comes next.

- Success as defined by you: We’ll provide the tools and flexibility, so you can make a meaningful impact, your way.

- Transformative leadership: We’ll give you the insights, coaching and confidence to be the leader the world needs.

- Diverse and inclusive culture: You’ll be embraced for who you are and empowered to use your voice to help others find theirs.
    """
    print(compute_fit_score(text_cv, text_cv))