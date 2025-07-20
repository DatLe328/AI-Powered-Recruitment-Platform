import json
import glob
import os, spacy

def remove_overlaps(entities):
    entities = sorted(entities, key=lambda x: (x[0], -x[1]))
    clean = []
    prev_end = -1
    for start, end, label in entities:
        if start >= prev_end:
            clean.append((start, end, label))
            prev_end = end
    return clean
def align_entity_to_tokens(text, entities, nlp):
    doc = nlp.make_doc(text)
    aligned = []
    for start, end, label in entities:
        token_start = None
        token_end = None
        for token in doc:
            if token.idx <= start < token.idx + len(token):
                token_start = token.idx
            if token.idx < end <= token.idx + len(token):
                token_end = token.idx + len(token)
        if token_start is not None and token_end is not None:
            aligned.append((token_start, token_end, label))
    return aligned
data_dir = "data/"
TRAIN_DATA = []
LIMIT = 50  # số lượng CV muốn đọc
skill_names = set()
count = 0
nlp = spacy.load("en_core_web_sm")
for filepath in glob.glob(os.path.join(data_dir, "cv *_annotated.json")):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    text = data["text"]
    entities = []
    for start, end, label in data["annotations"]:
        if "SKILL" in label:
            entities.append((start, end, "SKILL"))
    if entities:
        entities = remove_overlaps(entities)
        # entities = align_entity_to_tokens(text, entities, nlp)
        TRAIN_DATA.append((text, {"entities": entities}))
        count += 1
        for start, end, label in entities:
            skill_text = text[start:end].strip()
            skill_names.add(skill_text)
    if count >= LIMIT:
        break


print(f"Đã load {count} CV.")
print(f"Có tổng cộng {len(skill_names)} kỹ năng khác nhau.\n")
print("Một số kỹ năng:")
for skill in list(skill_names)[:20]:  # in thử 20 kỹ năng
    print(skill)


from spacy.util import minibatch, compounding
from spacy.training import Example
import random

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("SKILL")
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
optimizer = nlp.begin_training()

for itn in range(20):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for text, annots in batch:
            text = clean_text(text)
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annots))
        nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)
    print(f"Iteration {itn} Losses: {losses}")

# nlp.to_disk("skill_ner_model2")