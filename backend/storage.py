"""Image storage behind a tiny interface. The demo uses the local folder; an S3 class with the same
methods can replace it later without touching the API code."""
from pathlib import Path

EXTS = (".jpg", ".jpeg", ".png", ".webp")


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("key outside storage root")
        return p

    def save(self, key: str, data: bytes) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def url(self, key: str) -> str:
        return f"/assets/{key}"

    def find(self, stem_key: str):
        """Key of the first existing file named stem_key + any known image extension, else None."""
        for ext in EXTS:
            if self.exists(stem_key + ext):
                return stem_key + ext
        return None
