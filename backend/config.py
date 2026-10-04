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
    panel_assign_dist: float = 0.6
    photos_dir: Path = ROOT / "VEO Images"        # the real cube-face photos + cameras.json (not in git); image method needs it
    thumbs_dir: Path = ROOT / "data" / "thumbs"
    public_base_url: str = "http://localhost:8000"  # prefix for the picture/PDF links inside exports (Matterport/manual sheet)
    cors_origins: tuple = ("*",)                   # browser frontends (Lovable) need CORS
    e57_results: Path = ROOT / "outputs" / "physical_tags.csv"   # what the E57 pipeline already found (no upload needed)   # metres: a detected panel is matched to the nearest cabinet nameplate within this distance (H01's is 0.17 m off-centre)


def load_settings() -> Settings:
    return Settings(
        veo_db=_path("VEO_DB", "data/veo.db"),
        knowledge_db=_path("KNOWLEDGE_DB", "data/knowledge.db"),
        docs_dir=_path("DOCS_DIR", "docs"),
        assets_dir=_path("ASSETS_DIR", "assets"),
        site=os.environ.get("VEO_SITE", "eHouse"),
        conf_min=float(os.environ.get("CONF_MIN", "0.5")),
        max_assign_dist=float(os.environ.get("MAX_ASSIGN_DIST", "2.0")),
        panel_assign_dist=float(os.environ.get("PANEL_ASSIGN_DIST", "0.6")),
        photos_dir=_path("PHOTOS_DIR", "VEO Images"),
        thumbs_dir=_path("THUMBS_DIR", "data/thumbs"),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
        cors_origins=tuple(o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")),
        e57_results=_path("E57_RESULTS", "outputs/physical_tags.csv"),
    )
