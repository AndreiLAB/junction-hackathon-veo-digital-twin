"""Retrieval test: each question must return a chunk containing the expected phrase (taken from the manual text) in the top K.

    python scripts/kb_eval.py [--db data/knowledge.db] [-k 3]
"""
import argparse
import sqlite3
import sys

sys.path.insert(0, __file__.rsplit("scripts", 1)[0] + "scripts")
from kb_search import search  # noqa: E402

# (question, model filter, phrase that must appear in a returned chunk)
CASES = [
    ("What are the three protection indicators above the display?", "ABB 615", "Ready, Start/ Pickup and Trip"),
    ("What does the green Ready LED do when an internal relay fault is detected?", "ABB 615", "green Ready LED begins to flash"),
    ("What is the default rated frequency?", "ABB 615", "Rated frequency of"),
    ("What should the binary input threshold voltage be set to relative to auxiliary voltage?", "ABB 615", "70% of the nominal auxiliary voltage"),
    ("Which DC auxiliary voltages are supported?", "ABB 615", "24, 48, 60, 110, 125, 220 or 250 V DC"),
    ("Can LED indication control be used for tripping?", "ABB 615", "should never be used for tripping purposes"),
    ("What does the Ready LED do in IED test mode?", "ABB 615", "green Ready LED flashes"),
    ("What is the rated voltage of the UniGear ZS2 switchgear?", "UniGear ZS2", "Rated voltage (Ur)"),
    ("How is the circuit breaker moved between test and service positions?", "UniGear ZS2", "Test and Service positions"),
    ("What insulation resistance is required for the medium voltage circuit?", "VD4", "at least 50"),
    ("Which device signals whether the closing spring is charged?", "VD4", "closing spring charged/discharged"),
    ("What is the capital of Finland?", None, None),   # must return nothing relevant ("not found" behaviour)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/knowledge.db")
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    ok = 0
    scored = [c for c in CASES if c[2]]
    for q, model, phrase in CASES:
        hits = search(con, q, a.k, model)
        if phrase is None:
            print(f"[info] {q!r}: {len(hits)} hits returned (top: {hits[0]['section_path'] if hits else 'none'})")
            continue
        rank = next((i + 1 for i, h in enumerate(hits) if phrase.lower() in h["text"].lower()), None)
        ok += rank is not None
        top = hits[0] if hits else None
        where = f"rank {rank}" if rank else "MISS"
        print(f"[{where:7}] {q}\n          top: {top['document'][:14]} p.{top['page_start']} {top['section_path'][-70:]}" if top
              else f"[MISS   ] {q}  (no hits)")
    print(f"\n{ok}/{len(scored)} found in top {a.k}")
    return 0 if ok == len(scored) else 1


if __name__ == "__main__":
    sys.exit(main())
