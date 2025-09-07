#!/usr/bin/env python3
"""
Make a tiny, deterministic AML/KYC curriculum dataset (50 items total) for smoke tests.
No external API calls. Schema matches the training code:
{"instruction","input","output","cot","difficulty","label"}

Writes to:
  tests/data/easy.jsonl
  tests/data/medium.jsonl
  tests/data/hard.jsonl
  tests/data/train.jsonl
  tests/data/val.jsonl   (balanced ~30% of train, capped at 100)
"""

from __future__ import annotations
import os, json, random
from typing import List, Dict, Tuple
from dataclasses import dataclass

random.seed(1234)

OUT_DIR = os.path.join("tests", "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Simple name/locale pools for variation
PERSON_FIRST = {
    "latin": ["Juan", "Maria", "David", "Elena", "Marco", "Sofia"],
    "cyrillic": ["Ivan", "Olga", "Dmitri", "Nadia", "Alexei", "Irina"],
    "arabic": ["Omar", "Layla", "Yusuf", "Maha", "Karim", "Fatima"],
    "ch-translit": ["Zhang", "Wang", "Li", "Liu", "Chen", "Zhao"],
}
PERSON_LAST = {
    "latin": ["Garcia", "Rossi", "Muller", "Martin", "Dubois", "Silva"],
    "cyrillic": ["Petrov", "Sidorov", "Smirnov", "Kuznetsov", "Volkova", "Popov"],
    "arabic": ["Al-Masri", "Haddad", "Al-Sayed", "Nasser", "Aziz", "Rahman"],
    "ch-translit": ["Wei", "Hao", "Xin", "Qiang", "Tao", "Feng"],
}
COUNTRIES = {
    "latin": ["Spain", "Italy", "Portugal"],
    "cyrillic": ["Russia", "Ukraine", "Belarus"],
    "arabic": ["Egypt", "Jordan", "UAE"],
    "ch-translit": ["China", "Singapore", "Malaysia"],
}
ORG_SUFFIX = ["Ltd", "LLC", "S.A.", "GmbH", "PLC"]
HIGH_RISK = ["Iran", "North Korea", "Syria", "Sudan"]

def pick_locale() -> str:
    return random.choice(list(PERSON_FIRST.keys()))

def mk_person(locale: str) -> Tuple[str, str]:
    fn = random.choice(PERSON_FIRST[locale])
    ln = random.choice(PERSON_LAST[locale])
    return fn, ln

def mk_org(locale: str) -> str:
    stem = random.choice(["Global", "Eastern", "Pacific", "Nordic", "Atlas", "Horizon"])
    tail = random.choice(["Trading", "Holdings", "Capital", "Logistics", "Mining", "Tech"])
    suffix = random.choice(ORG_SUFFIX)
    return f"{stem} {tail} {suffix}"

def dob_str(year: int) -> str:
    # Basic YYYY-MM banded
    return f"{year:04d}-??-??"

def mk_case(idx: int, difficulty: str, label: str) -> Dict:
    locale = pick_locale()
    is_person = random.random() < 0.65
    if is_person:
        fn, ln = mk_person(locale)
        cand_name = f"{fn} {ln}"
        watch_name = cand_name if label == "match" else f"{fn} {ln}{random.choice(['', 'a', 'ov', '-Lee'])}"
        # DOB bands — clearer for easy, fuzzier for hard
        base_year = random.choice(range(1965, 1998))
        cand_dob = dob_str(base_year + (0 if difficulty == "easy" else random.choice([-1, 0, +1])))
        watch_dob = dob_str(base_year)
        country_cand = random.choice(COUNTRIES[locale])
        country_watch = country_cand if label != "no_match" else random.choice(HIGH_RISK if random.random() < 0.4 else sum(COUNTRIES.values(), []))
        id_hint = f"NatID:{random.randint(10000000,99999999)}"
        translit_note = "Transliteration: none" if difficulty == "easy" else f"Transliteration: {locale}"
        input_txt = (
            f"Candidate: {cand_name}, DOB≈{cand_dob}, Country: {country_cand}, {id_hint}. "
            f"Watchlist: {watch_name}, DOB≈{watch_dob}, Country: {country_watch}. {translit_note}. "
            f"Aliases: {watch_name if difficulty!='hard' else watch_name.replace(' ', '')}."
        )
        # CoT short & explicit
        cot = (
            f"Compare name and DOB bands. Name similarity high. DOB close. "
            f"Countries {'match' if country_cand==country_watch else 'differ'}. "
            f"ID format plausible."
        )
        # Decision logic
        if label == "match":
            output = "MATCH — name + DOB band align; country acceptable; aliases consistent."
        elif label == "no_match":
            output = "NO_MATCH — country differs materially and aliases do not support identity."
        else:
            output = "EDGE — conflicting country or partial DOB; insufficient certainty."
    else:
        # Organization-style case
        org_c = mk_org(locale)
        org_w = org_c if label == "match" else mk_org(locale)
        lei = f"LEI-{random.randint(100000,999999)}"
        parent = mk_org(locale) if difficulty != "easy" else ""
        country_cand = random.choice(COUNTRIES[locale])
        country_watch = country_cand if label != "no_match" else random.choice(HIGH_RISK)
        rel = "" if difficulty == "easy" else f"Parent/UBO: {parent}. "
        input_txt = (
            f"Candidate Org: {org_c}, Country: {country_cand}, LEI: {lei}. "
            f"Watchlist Org: {org_w}, Country: {country_watch}. {rel}"
            f"Aliases include suffix variants."
        )
        cot = (
            "Check canonical org stem vs suffix variants; compare LEI and country. "
            "Ownership links considered if present."
        )
        if label == "match":
            output = "MATCH — canonical stem + LEI + country alignment indicate same entity."
        elif label == "no_match":
            output = "NO_MATCH — different canonical stem and high-risk country mismatch."
        else:
            output = "EDGE — possible parent/subsidiary relation; identity uncertain."

    # Make difficulty-specific tiny perturbations
    if difficulty == "medium":
        input_txt += " Minor noise: alternate spelling present."
    elif difficulty == "hard":
        input_txt += " Mixed scripts/transliteration and partial IDs."

    rec = {
        "instruction": "Decide if the candidate matches the watchlist entity and give a one-line rationale.",
        "input": input_txt,
        "cot": cot,
        "output": output,
        "difficulty": difficulty,
        "label": label,
    }
    return rec

def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    # Plan: 50 total -> easy:20, medium:18, hard:12
    plan = [
        ("easy", 20),
        ("medium", 18),
        ("hard", 12),
    ]
    labels = ["match", "no_match", "edge"]

    all_rows: List[Dict] = []
    per_tier: dict = {tier: [] for tier, _ in plan}
    idx = 0
    for tier, n in plan:
        for _ in range(n):
            idx += 1
            # rotate labels to keep balance-ish
            label = labels[idx % 3]
            rec = mk_case(idx, tier, label)
            # Ensure minimum lengths (schema expectations)
            assert len(rec["instruction"]) >= 5
            assert len(rec["input"]) >= 10
            assert len(rec["cot"]) >= 5
            assert len(rec["output"]) >= 3
            per_tier[tier].append(rec)
            all_rows.append(rec)

    # Write per-tier
    for tier, rows in per_tier.items():
        write_jsonl(os.path.join(OUT_DIR, f"{tier}.jsonl"), rows)

    # Train = all 50
    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), all_rows)

    # Val: ~30% balanced by (difficulty,label) buckets, capped at 100
    buckets = {}
    for r in all_rows:
        buckets.setdefault((r["difficulty"], r["label"]), []).append(r)
    val = []
    target = min(100, max(1, int(round(0.3 * len(all_rows)))))
    # simple round-robin across buckets
    keys = list(buckets.keys())
    i = 0
    while len(val) < target:
        k = keys[i % len(keys)]
        if buckets[k]:
            val.append(buckets[k].pop(0))
        i += 1
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)

    print(f"Wrote boilerplate dataset to {OUT_DIR}:")
    print(" - easy.jsonl, medium.jsonl, hard.jsonl")
    print(" - train.jsonl (50 items), val.jsonl (~15 items)")

if __name__ == "__main__":
    main()
