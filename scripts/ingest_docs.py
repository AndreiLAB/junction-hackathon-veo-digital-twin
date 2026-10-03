"""Convert manuals under docs/<folder>/ into section-based Markdown + a searchable SQLite knowledge base.

Usage:
    python scripts/ingest_docs.py [--docs docs] [--md docs_md] [--db data/knowledge.db] [--only NAME] [--max-pages N]

Pipeline per PDF (PyMuPDF only, no layout models):
  1. headings: from the PDF bookmarks (TOC) when present, else from font size
  2. running headers/footers removed (repeating text in the page margins)
  3. tables extracted as Markdown tables and kept whole
  4. text split into chunks of ~MAX_CHARS inside each section, never inside a table row
  5. every chunk stores its breadcrumb (section path) and page range for citations

Outputs: docs_md/<folder>/<stem>.md (readable full text) and data/knowledge.db
(documents, chunks, chunks_fts). Search with scripts/kb_search.py.
"""
import argparse
import collections
import hashlib
import re
import sqlite3
import sys
import time
from pathlib import Path

import pymupdf

MODEL_BY_FOLDER = {"abb_615": "ABB 615", "unigear_zs2": "UniGear ZS2", "vd4": "VD4"}
MAX_CHARS = 2200          # ~500-600 tokens per chunk
MIN_CHARS = 40            # chunks shorter than this carry no information
MARGIN_TOP, MARGIN_BOTTOM = 0.07, 0.07
SKIP_TITLES = re.compile(r"^(table of )?contents$|^list of (figures|tables)$", re.I)
BULLET = re.compile(r"^(•|▪|●|-|–|\d+[.)]|[a-z][.)])\s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, model TEXT, doc_type TEXT, doc_number TEXT,
    file_path TEXT NOT NULL UNIQUE, md_path TEXT, sha256 TEXT, pages INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, section_path TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
    kind TEXT NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(section_path, text, tokenize='porter unicode61');
"""


# ---------- helpers ----------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip().lower()


def strip_number(s: str) -> str:
    return re.sub(r"^([A-Z]\.)?[\d.]*\d\.?\s+", "", s.strip())


def margin_key(s: str) -> str:
    return re.sub(r"\d+", "#", norm(s))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doc_meta(stem: str):
    num = re.search(r"2NGA\d{6}|(?<!\d)\d{6}(?!\d)", stem)
    low = stem.lower()
    dtype = "technical manual" if re.search(r"(^|[_ -])tm([_ -]|$)", low) else \
        "instruction manual" if "instruction manual" in low else \
        "manual" if low.startswith("ma_") else None
    return (num.group(0) if num else None), dtype


# ---------- page parsing ----------
def line_records(page, tables_bbox):
    """Flat list of text lines (excluding tables) with size/bold/block id/y."""
    out = []
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        if b["type"] != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in tables_bbox):
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"]).replace("�", "'").strip()
            main = max(spans, key=lambda s: len(s["text"]))
            bold = bool(main["flags"] & 16) or "bold" in main["font"].lower()
            out.append({"text": text, "size": round(main["size"] * 2) / 2, "bold": bold,
                        "block": bi, "y": ln["bbox"][1], "x": ln["bbox"][0], "cy": cy})
    out.sort(key=lambda r: (round(r["y"]), r["x"]))
    return out


def prepass(doc, step_pages=40):
    """Body font size and repeating margin text, sampled across the document."""
    sizes, margin = collections.Counter(), collections.Counter()
    idx = list(range(0, len(doc), max(1, len(doc) // step_pages)))
    for i in idx:
        p = doc[i]
        h = p.rect.height
        for r in line_records(p, []):
            sizes[r["size"]] += len(r["text"])
            if r["cy"] < h * MARGIN_TOP or r["cy"] > h * (1 - MARGIN_BOTTOM):
                margin[margin_key(r["text"])] += 1
    body = sizes.most_common(1)[0][0] if sizes else 10
    repeated = {k for k, v in margin.items() if v >= max(2, 0.2 * len(idx))}
    return body, repeated


class Builder:
    """Accumulates sections -> units (paragraphs/tables) with page numbers."""

    def __init__(self):
        self.sections = []          # dict(path=[...], level, units=[(kind, text, page)])
        self.stack = []             # [(level, title)]
        self._new([], 0)

    def _new(self, path, level):
        self.sections.append({"path": list(path), "level": level, "units": []})

    def heading(self, level, title):
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        self._new([t for _, t in self.stack], level)

    def add(self, kind, text, page):
        if text.strip():
            self.sections[-1]["units"].append((kind, text.strip(), page))


def paragraphs(lines):
    """Join lines of one block into paragraphs; keep bullets on their own line, fix hyphen breaks."""
    out, cur = [], ""
    for ln in lines:
        t = ln["text"]
        if BULLET.match(t) and cur:
            out.append(cur)
            cur = t
        elif cur.endswith("-") and t[:1].islower():
            cur = cur[:-1] + t
        else:
            cur = (cur + " " + t).strip()
    if cur:
        out.append(cur)
    return out


def parse_doc(doc, max_pages):
    toc = doc.get_toc()
    body, repeated = prepass(doc)
    sizes_rank = None
    if not toc:
        # heading levels from font sizes larger than body
        cnt = collections.Counter()
        for i in range(0, len(doc), max(1, len(doc) // 40)):
            for r in line_records(doc[i], []):
                if r["size"] >= body + 1.0 and len(r["text"]) < 90:
                    cnt[r["size"]] += 1
        sizes_rank = {s: i + 1 for i, s in enumerate(sorted(cnt, reverse=True)[:3])}
    toc_by_page = collections.defaultdict(list)
    for lvl, title, pg in toc:
        toc_by_page[pg].append((min(lvl, 4), title.strip()))

    b = Builder()
    unmatched = 0
    n = min(len(doc), max_pages or len(doc))
    t0 = time.time()
    for pi in range(n):
        page = doc[pi]
        pno = pi + 1
        h = page.rect.height
        tabs = []
        try:
            tabs = page.find_tables().tables
        except Exception:
            pass
        lines = [r for r in line_records(page, [t.bbox for t in tabs])
                 if not ((r["cy"] < h * MARGIN_TOP or r["cy"] > h * (1 - MARGIN_BOTTOM))
                         and margin_key(r["text"]) in repeated)]
        entries = list(toc_by_page.get(pno, []))
        # sequential walk: emit headings, group the rest by block
        pending, last_block = [], None

        def flush():
            nonlocal pending
            for para in paragraphs(pending):
                b.add("text", para, pno)
            pending = []

        i = 0
        while i < len(lines):
            r = lines[i]
            heading = None
            if toc:
                for ei, (lvl, title) in enumerate(entries):
                    want = norm(strip_number(title))
                    got, j = norm(r["text"]), i
                    while want.startswith(got) and got != want and j + 1 < len(lines) and len(got) < len(want):
                        j += 1
                        got = norm(got + " " + lines[j]["text"])
                    if got == want or (len(want) > 10 and got.startswith(want)):
                        heading = (lvl, title.replace("\xa0", " "), j)
                        entries.pop(ei)
                        break
            elif r["size"] in sizes_rank and len(r["text"]) < 90 and not r["text"].endswith((".", ",")):
                heading = (sizes_rank[r["size"]], r["text"], i)
            if heading:
                lvl, title, j = heading
                # drop a bare number line that preceded the heading text ("3.25.2.1" / title)
                if pending and re.fullmatch(r"[A-Z]?[\d.]+", pending[-1]["text"]):
                    pending.pop()
                flush()
                b.heading(lvl, title)
                i = j + 1
                last_block = None
                continue
            if last_block is not None and r["block"] != last_block:
                flush()
            last_block = r["block"]
            pending.append(r)
            i += 1
        flush()
        for t in sorted(tabs, key=lambda t: t.bbox[1]):
            try:
                md = t.to_markdown().strip()
            except Exception:
                continue
            if md:
                b.add("table", md.replace("\n\n", "\n"), pno)
        for lvl, title in entries:            # TOC entries whose text wasn't found on the page
            unmatched += 1
            b.heading(lvl, title.replace("\xa0", " "))
        if pno % 100 == 0 or pno == n:
            print(f"  page {pno}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    return b.sections, unmatched, n


# ---------- chunking / output ----------
def split_table(md, limit):
    rows = md.split("\n")
    head, body = rows[:2], rows[2:]
    parts, cur = [], []
    for r in body:
        if cur and len("\n".join(head + cur + [r])) > limit:
            parts.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def chunks_of(sections):
    out = []
    for s in sections:
        if not s["units"]:
            continue
        path = " > ".join(s["path"]) or "(front matter)"
        if SKIP_TITLES.match(norm(s["path"][0] if s["path"] else "")):
            continue
        cur, kind, pages = [], "text", []

        def emit():
            nonlocal cur, kind, pages
            text = "\n\n".join(cur).strip()
            if len(text) >= MIN_CHARS:
                out.append({"section_path": path, "page_start": min(pages), "page_end": max(pages),
                            "kind": kind, "text": text})
            cur, kind, pages = [], "text", []

        for k, text, pg in s["units"]:
            pieces = split_table(text, MAX_CHARS) if k == "table" and len(text) > MAX_CHARS else [text]
            for piece in pieces:
                if cur and len("\n\n".join(cur)) + len(piece) > MAX_CHARS:
                    emit()
                cur.append(piece)
                pages.append(pg)
                if k == "table":
                    kind = "table" if len(cur) == 1 else "mixed"
        emit()
    return out


def write_md(path, meta, sections):
    lines = ["---"] + [f"{k}: {v}" for k, v in meta.items() if v is not None] + ["---", ""]
    last_pg = None
    for s in sections:
        if s["path"]:
            lines += ["#" * min(s["level"], 6) + " " + s["path"][-1], ""]
        for k, text, pg in s["units"]:
            if pg != last_pg:
                lines.append(f"<!-- p.{pg} -->")
                last_pg = pg
            lines += [text, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def ingest(pdf: Path, root: Path, md_dir: Path, con, max_pages):
    rel = pdf.relative_to(root).as_posix()
    digest = sha256(pdf) + (f":{max_pages}" if max_pages else "")
    row = con.execute("SELECT id, sha256 FROM documents WHERE file_path=?", (rel,)).fetchone()
    if row and row[1] == digest:
        return "unchanged"
    if row:
        ids = [r[0] for r in con.execute("SELECT id FROM chunks WHERE document_id=?", (row[0],))]
        con.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        con.execute("DELETE FROM chunks WHERE document_id=?", (row[0],))
        con.execute("DELETE FROM documents WHERE id=?", (row[0],))

    doc = pymupdf.open(pdf)
    sections, unmatched, n = parse_doc(doc, max_pages)
    chunks = chunks_of(sections)
    folder = pdf.parent.name
    number, dtype = doc_meta(pdf.stem)
    model = MODEL_BY_FOLDER.get(folder)
    md_path = md_dir / folder / (pdf.stem + ".md")
    write_md(md_path, {"title": pdf.stem, "model": model, "doc_type": dtype, "doc_number": number,
                       "source_pdf": rel, "pages": len(doc)}, sections)
    cur = con.execute(
        "INSERT INTO documents(title, model, doc_type, doc_number, file_path, md_path, sha256, pages) VALUES (?,?,?,?,?,?,?,?)",
        (pdf.stem, model, dtype, number, rel, md_path.as_posix(), digest, len(doc)))
    for i, c in enumerate(chunks):
        r = con.execute(
            "INSERT INTO chunks(document_id, ord, section_path, page_start, page_end, kind, text) VALUES (?,?,?,?,?,?,?)",
            (cur.lastrowid, i, c["section_path"], c["page_start"], c["page_end"], c["kind"], c["text"]))
        con.execute("INSERT INTO chunks_fts(rowid, section_path, text) VALUES (?,?,?)",
                    (r.lastrowid, c["section_path"], c["text"]))
    con.commit()
    avg = sum(len(c["text"]) for c in chunks) // max(1, len(chunks))
    tabs = sum(1 for c in chunks if c["kind"] != "text")
    return (f"{n}/{len(doc)} pages -> {len(sections)} sections, {len(chunks)} chunks "
            f"(avg {avg} chars, {tabs} with tables), {unmatched} TOC entries not found on their page")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--md", default="docs_md")
    ap.add_argument("--db", default="data/knowledge.db")
    ap.add_argument("--only", help="substring of the PDF file name")
    ap.add_argument("--max-pages", type=int, help="testing: only the first N pages")
    a = ap.parse_args()

    root = Path(a.docs)
    pdfs = [p for p in sorted(root.rglob("*.pdf")) if not a.only or a.only.lower() in p.name.lower()]
    if not pdfs:
        print(f"No matching PDFs under {root}", file=sys.stderr)
        return 1
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    failed = 0
    for pdf in pdfs:
        print(f"{pdf.relative_to(root)}", flush=True)
        try:
            print("  ->", ingest(pdf, root, Path(a.md), con, a.max_pages), flush=True)
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e!r}", file=sys.stderr)
    print(f"done: {len(pdfs) - failed}/{len(pdfs)} ok -> {a.db}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
