import spacy
from spacy.matcher import PhraseMatcher
from test import SkillExtractor
from your_skill_db import SKILL_DB  # dict Lightcast/EMSI

nlp = spacy.load("en_core_web_sm")
extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

text = "We use SQL. Experience with Java Stored Procedure (SQL) is a plus, not to mention NLP."
res = extractor.annotate(text, tresh=0.5)
print(res["results"]["full_matches"])
print(res["results"]["ngram_scored"])
