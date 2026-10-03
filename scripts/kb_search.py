"""Search the manuals knowledge base. Usable as a CLI and as a library (search()).

    python scripts/kb_search.py "READY LED meaning" [--model "ABB 615"] [-k 5] [--db data/knowledge.db]

Returns the best-matching chunks with document, section path and page range so
callers (the /ask endpoint, an LLM tool) can cite them. Returns [] when nothing matches.
"""
import argparse
import math
import re
import sqlite3
import sys


STOP = set("a an the of in on at to for from by with and or is are was were be been what which who whom how when where why does do did "
           "can could should would will may this that these those it its as into than then there their they you your we our i".split())
MIN_COVERAGE = 0.6   # share of the question's content words a chunk must contain, else it is not an answer


def terms(q: str, model=None):
    skip = STOP | ({w.lower() for w in re.findall(r"[A-Za-z0-9_]+", model)} if model else set())
    return [w for w in re.findall(r"[A-Za-z0-9_]+", q) if w.lower() not in skip]


def fts_query(q: str, model=None) -> str:
    return " OR ".join(f'"{w}"' for w in terms(q, model))


def search(con, q, k=5, model=None, doc_ids=None):
    fq = fts_query(q, model)
    if not fq:
        return []
    sql = """SELECT c.id, d.id, d.title, d.model, c.section_path, c.page_start, c.page_end, c.kind, c.text,
                    bm25(chunks_fts, 4.0, 1.0) AS score
             FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN documents d ON d.id = c.document_id
             WHERE chunks_fts MATCH ?"""
    args = [fq]
    if model:
        sql += " AND d.model = ?"
        args.append(model)
    if doc_ids:
        sql += f" AND d.id IN ({','.join('?' * len(doc_ids))})"
        args += list(doc_ids)
    sql += " ORDER BY score LIMIT ?"
    args.append(max(k * 10, 30))
    keys = ["chunk_id", "document_id", "document", "model", "section_path", "page_start", "page_end", "kind", "text", "score"]
    rows = [dict(zip(keys, r)) for r in con.execute(sql, args)]
    words = {w.lower() for w in terms(q, model)}
    need = max(1, math.ceil(MIN_COVERAGE * len(words)))
    kept = []
    for r in rows:
        have = {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", r["section_path"] + " " + r["text"])}
        if len(words & have) >= need:
            kept.append(r)
    return kept[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--db", default="data/knowledge.db")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    hits = search(con, a.query, a.k, a.model)
    if not hits:
        print("not found")
        return 0
    for h in hits:
        pg = f"p.{h['page_start']}" if h["page_start"] == h["page_end"] else f"pp.{h['page_start']}-{h['page_end']}"
        print(f"[{h['document']} {pg}] {h['section_path']}  (score {h['score']:.2f})")
        print(h["text"][:600].replace("\n", "\n    "))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
