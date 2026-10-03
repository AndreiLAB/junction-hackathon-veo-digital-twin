"""Backend settings. Every path can be overridden with an environment variable (no hardcoded absolute paths)."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, ROOT / default))


@dataclass
class Settings:
    veo_db: Path            # tags + devices (created on first start)
    knowledge_db: Path      # manuals: documents, chunks, FTS (built by scripts/ingest_docs.py)
    docs_dir: Path          # the PDFs; documents.file_path is relative to this
    assets_dir: Path        # images served under /assets
    site: str               # site name returned by GET /tags
    conf_min: float         # detections below this are flagged needs_review
    max_assign_dist: float  # metres: a device farther than this from every cabinet stays unassigned


def load_settings() -> Settings:
    return Settings(
        veo_db=_path("VEO_DB", "data/veo.db"),
        knowledge_db=_path("KNOWLEDGE_DB", "data/knowledge.db"),
        docs_dir=_path("DOCS_DIR", "docs"),
        assets_dir=_path("ASSETS_DIR", "assets"),
        site=os.environ.get("VEO_SITE", "eHouse"),
        conf_min=float(os.environ.get("CONF_MIN", "0.5")),
        max_assign_dist=float(os.environ.get("MAX_ASSIGN_DIST", "2.0")),
    )
