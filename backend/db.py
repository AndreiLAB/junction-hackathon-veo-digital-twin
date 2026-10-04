"""SQLite for tags and devices. Manuals live in a separate read-only database (knowledge.db)."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, folder TEXT NOT NULL DEFAULT 'Cabinets',
    x REAL, y REAL, z REAL, confidence REAL, needs_review INTEGER NOT NULL DEFAULT 0,
    read_as TEXT, sightings INTEGER, evidence_crop TEXT,
    doc_models TEXT NOT NULL DEFAULT '[]',
    image_is_placeholder INTEGER NOT NULL DEFAULT 1,
    created_by TEXT NOT NULL DEFAULT 'seed',
    panel_model TEXT, panel_source TEXT, panel_confidence REAL,
    e57_found INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tag_id TEXT REFERENCES tags(id),
    class TEXT NOT NULL, type TEXT NOT NULL, model TEXT,
    confidence REAL, x REAL, y REAL, z REAL,
    ocr_text TEXT NOT NULL DEFAULT '[]', ocr_conf REAL,
    source_image TEXT, box TEXT, crop_path TEXT,
    needs_review INTEGER NOT NULL DEFAULT 0, review_reasons TEXT NOT NULL DEFAULT '[]',
    created_by TEXT NOT NULL DEFAULT 'auto'
);
"""


def connect(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init(con: sqlite3.Connection, seed: Path) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(tags)")}
    for col, typ in (("panel_model", "TEXT"), ("panel_source", "TEXT"), ("panel_confidence", "REAL"), ("e57_found", "INTEGER NOT NULL DEFAULT 0")):   # databases created before the panel fields
        if col not in have:
            con.execute(f"ALTER TABLE tags ADD COLUMN {col} {typ}")
    for c in json.loads(Path(seed).read_text(encoding="utf-8")):
        pm = c.get("panel_model")
        con.execute("INSERT OR IGNORE INTO tags(id, name, doc_models, panel_model, panel_source) VALUES (?,?,?,?,?)",
                    (c["id"], c["name"], json.dumps(c["doc_models"]), pm, "assumed" if pm else None))
        if pm:   # older seeded rows: add the (assumed) panel type and the datasheet link without touching detected evidence
            con.execute("UPDATE tags SET panel_model=?, panel_source='assumed' WHERE id=? AND panel_model IS NULL", (pm, c["id"]))
            con.execute("UPDATE tags SET doc_models=? WHERE id=? AND doc_models='[]' AND created_by='seed'", (json.dumps(c["doc_models"]), c["id"]))
    con.commit()
