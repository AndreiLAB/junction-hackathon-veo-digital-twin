"""Read-only access to the manuals knowledge base built by scripts/ingest_docs.py."""
import sqlite3
from pathlib import Path

from scripts.kb_search import search  # noqa: F401  (re-exported for the API)

DOC_COLS = "id, title, model, doc_type, doc_number, file_path, pages"


def open_ro(path: Path):
    if not Path(path).is_file():
        return None
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def doc_dict(r) -> dict:
    return {"id": r["id"], "title": r["title"], "model": r["model"], "doc_type": r["doc_type"],
            "doc_number": r["doc_number"], "pages": r["pages"], "url": f"/documents/{r['id']}/file"}


def documents_for_models(con, models) -> list:
    models = list(models)
    if con is None or not models:
        return []
    q = ",".join("?" * len(models))
    rows = con.execute(f"SELECT {DOC_COLS} FROM documents WHERE model IN ({q}) ORDER BY id", models)
    return [doc_dict(r) for r in rows]


def all_documents(con, model=None) -> list:
    if con is None:
        return []
    sql, args = f"SELECT {DOC_COLS} FROM documents", []
    if model:
        sql += " WHERE model = ?"
        args.append(model)
    return [doc_dict(r) for r in con.execute(sql + " ORDER BY id", args)]


def get_document(con, doc_id: int):
    if con is None:
        return None
    return con.execute(f"SELECT {DOC_COLS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
