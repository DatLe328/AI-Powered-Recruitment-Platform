import json
import collections
import os

SKILL_PATH = '../data/processed/skills_processed.json'
TOKEN_PATH = '../data/processed/token_dist.json'

def main():
    if not os.path.isfile(SKILL_PATH):
        raise FileNotFoundError(f"File not found: {SKILL_PATH}")
    with open(SKILL_PATH, 'r', encoding='utf-8') as f:
        skill_db = json.load(f)

    words = [
        w
        for v in skill_db.values()
        if v.get('skill_len', 0) > 1
        for w in (v.get('skill_stemmed', '') or '').split()
        if w
    ]

    token_dist = dict(collections.Counter(words))

    with open(TOKEN_PATH, 'w', encoding='utf-8') as f:
        json.dump(token_dist, f, ensure_ascii=False, indent=2)

    print(f"Token distribution saved to {TOKEN_PATH}. Total unique tokens: {len(token_dist)}")

if __name__ == "__main__":
    main()