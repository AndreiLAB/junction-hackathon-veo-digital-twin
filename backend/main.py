"""FastAPI backend: serves cabinet tags (with devices, documents, pictures) to the viewer.

Run from the repo root:  uvicorn backend.main:app --reload
"""
import json
import math
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, knowledge
from .config import ROOT, Settings, load_settings
from .storage import LocalStorage

# detector class -> (device type, manual model). nameplate is evidence for OCR, not a device.
DEVICE_CLASSES = {
    "abb_relion_615": ("ABB 615 protection relay", "ABB 615"),
    "relay_front": ("ABB 615 protection relay", "ABB 615"),
    "relay_rear": ("ABB 615 protection relay", "ABB 615"),
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class Position(BaseModel):
    x: float
    y: float
    z: float


class DetectionIn(BaseModel):
    model_config = {"populate_by_name": True}
    image: str = Field(description="source image name, e.g. scan_012_skybox_3")
    class_name: str = Field(alias="class", description="detector class, e.g. abb_relion_615")
    conf: float = Field(ge=0, le=1)
    box: Optional[List[float]] = Field(default=None, min_length=4, max_length=4,
                                       description="x, y, w, h in source-image pixels")
    ocr: List[str] = []
    ocr_conf: Optional[float] = Field(default=None, ge=0, le=1)
    position: Optional[Position] = Field(default=None, description="x, y, z if known; null leaves the device unassigned")


class CabinetIn(BaseModel):
    """One entry of Pragati's synthetic_pipeline/labels_out/tags.json."""
    tag: str
    name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    confidence: Optional[float] = None
    sightings: Optional[int] = None
    read_as: Optional[str] = None
    evidence_crop: Optional[str] = None


class TagPatch(BaseModel):
    name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    needs_review: Optional[bool] = None
    doc_models: Optional[List[str]] = None


class AskIn(BaseModel):
    question: str
    tag_id: Optional[str] = Field(default=None, description="limit the search to this cabinet's documents")
    model: Optional[str] = Field(default=None, description="limit to one manual model, e.g. 'ABB 615'")
    k: int = Field(default=5, ge=1, le=20)


def _pos(r):
    return {"x": r["x"], "y": r["y"], "z": r["z"]} if r["x"] is not None else None


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    s = settings or load_settings()
    con = db.connect(s.veo_db)
    db.init(con, ROOT / "backend" / "seed_cabinets.json")
    kb = knowledge.open_ro(s.knowledge_db)
    storage = LocalStorage(s.assets_dir)

    app = FastAPI(title="VEO360 auto-tagging backend", version="0.1.0")
    app.mount("/assets", StaticFiles(directory=s.assets_dir, check_dir=False), name="assets")

    # ---------- serializers ----------
    def device_out(r) -> dict:
        return {
            "id": r["id"], "tag_id": r["tag_id"], "type": r["type"], "class": r["class"], "model": r["model"],
            "confidence": r["confidence"], "needs_review": bool(r["needs_review"]),
            "review_reasons": json.loads(r["review_reasons"]), "position": _pos(r),
            "ocr_text": json.loads(r["ocr_text"]), "ocr_conf": r["ocr_conf"],
            "crop": storage.url(r["crop_path"]) if r["crop_path"] else None,
            "documents": knowledge.documents_for_models(kb, [r["model"]] if r["model"] else []),
            "source": {"image": r["source_image"], "box": json.loads(r["box"]) if r["box"] else None},
            "created_by": r["created_by"],
        }

    def tag_out(r) -> dict:
        key = storage.find(f"context/{r['id']}")
        devices = con.execute("SELECT * FROM devices WHERE tag_id = ? ORDER BY id", (r["id"],)).fetchall()
        return {
            "id": r["id"], "name": r["name"], "folder": r["folder"], "position": _pos(r),
            "confidence": r["confidence"], "needs_review": bool(r["needs_review"]),
            "image": storage.url(key) if key else None,
            "image_is_placeholder": bool(r["image_is_placeholder"]) if key else None,
            "read_as": r["read_as"], "sightings": r["sightings"], "evidence_crop": r["evidence_crop"],
            "documents": knowledge.documents_for_models(kb, json.loads(r["doc_models"])),
            "devices": [device_out(d) for d in devices],
            "created_by": r["created_by"],
        }

    def get_tag_row(tag_id: str):
        r = con.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
        if r is None:
            raise HTTPException(404, f"tag {tag_id} not found")
        return r

    # ---------- endpoints ----------
    @app.get("/health")
    def health():
        return {"status": "ok",
                "tags": con.execute("SELECT COUNT(*) FROM tags").fetchone()[0],
                "devices": con.execute("SELECT COUNT(*) FROM devices").fetchone()[0],
                "knowledge_base": kb is not None,
                "documents": len(knowledge.all_documents(kb))}

    @app.get("/tags")
    def list_tags(needs_review: Optional[bool] = None, folder: Optional[str] = None):
        sql, args = "SELECT * FROM tags WHERE 1=1", []
        if needs_review is not None:
            sql += " AND needs_review = ?"
            args.append(int(needs_review))
        if folder:
            sql += " AND folder = ?"
            args.append(folder)
        tags = [tag_out(r) for r in con.execute(sql + " ORDER BY rowid", args)]
        return {"site": s.site, "count": len(tags), "tags": tags}

    @app.get("/tags/{tag_id}")
    def get_tag(tag_id: str):
        return tag_out(get_tag_row(tag_id))

    @app.patch("/tags/{tag_id}")
    def patch_tag(tag_id: str, body: TagPatch):
        get_tag_row(tag_id)
        fields = body.model_dump(exclude_unset=True)
        if "doc_models" in fields:
            fields["doc_models"] = json.dumps(fields["doc_models"])
        if "needs_review" in fields:
            fields["needs_review"] = int(fields["needs_review"])
        if fields:
            con.execute(f"UPDATE tags SET {', '.join(f + ' = ?' for f in fields)} WHERE id = ?",
                        [*fields.values(), tag_id])
            con.commit()
        return tag_out(get_tag_row(tag_id))

    @app.put("/tags/{tag_id}/image")
    async def put_tag_image(tag_id: str, file: UploadFile = File(...), placeholder: bool = True):
        get_tag_row(tag_id)
        name = file.filename or ""
        ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        if ext not in IMAGE_EXTS:
            raise HTTPException(422, "image must be .jpg, .jpeg, .png or .webp")
        old = storage.find(f"context/{tag_id}")
        storage.save(f"context/{tag_id}{ext}", await file.read())
        if old and old != f"context/{tag_id}{ext}":
            storage.path(old).unlink()
        con.execute("UPDATE tags SET image_is_placeholder = ? WHERE id = ?", (int(placeholder), tag_id))
        con.commit()
        return tag_out(get_tag_row(tag_id))

    @app.post("/process")
    async def process_e57_file(file: UploadFile = File(...), high_accuracy: bool = False):
        """
        Receives an E57 file, runs the extraction, OCR, object detection, 3D raycasting,
        and aggregation pipeline, then stores the resulting assets into the database.
        """
        temp_path = ROOT / file.filename
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        import sys
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))
            
        from process_e57 import run_pipeline
        physical_tags = run_pipeline(str(temp_path), detector_type="mock", high_accuracy=high_accuracy)
        
        # Import generated physical tags into database
        updated, added, devices_added = 0, 0, 0
        
        # 1. Add all cubicles first
        for ptag in [p for p in physical_tags if p.asset_type == "cubicle"]:
            if con.execute("SELECT 1 FROM tags WHERE id = ?", (ptag.label,)).fetchone():
                updated += 1
            else:
                con.execute("INSERT INTO tags(id, name, created_by) VALUES (?,?,'pipeline')", (ptag.label, ptag.label))
                added += 1
                
            review = ptag.confidence is not None and ptag.confidence < s.conf_min
            con.execute("UPDATE tags SET x=?, y=?, z=?, confidence=?, sightings=?, needs_review=? WHERE id=?",
                        (ptag.x, ptag.y, ptag.z, ptag.confidence, ptag.observation_count, int(review), ptag.label))
        con.commit()
        
        # 2. Add equipment and link to nearest cubicle
        cabinets = con.execute("SELECT * FROM tags WHERE x IS NOT NULL").fetchall()
        for ptag in [p for p in physical_tags if p.asset_type == "equipment"]:
            dtype, model = DEVICE_CLASSES.get(ptag.label, ("Unknown", ptag.label))
            tag_id = None
            if cabinets:
                def dist(c):
                    return math.dist((c["x"], c["y"], c["z"]), (ptag.x, ptag.y, ptag.z))
                best = min(cabinets, key=dist)
                if dist(best) <= s.max_assign_dist:
                    tag_id = best["id"]
                    
            con.execute(
                "INSERT INTO devices(tag_id, class, type, model, confidence, x, y, z, created_by) VALUES (?,?,?,?,?,?,?,?,'pipeline')",
                (tag_id, ptag.label, dtype, model, ptag.confidence, ptag.x, ptag.y, ptag.z)
            )
            devices_added += 1
                        
        con.commit()
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except:
            pass
            
        return {
            "message": "E57 Processing Complete", 
            "physical_tags_found": len(physical_tags), 
            "cabinets_updated": updated, 
            "cabinets_added": added,
            "devices_added": devices_added
        }

    @app.post("/import/cabinets")
    def import_cabinets(items: List[CabinetIn]):
        """Load cabinet positions/confidence from find_labels.py's tags.json. Unknown ids are added."""
        updated, added = 0, 0
        for c in items:
            if con.execute("SELECT 1 FROM tags WHERE id = ?", (c.tag,)).fetchone():
                updated += 1
            else:
                con.execute("INSERT INTO tags(id, name, created_by) VALUES (?,?,'import')", (c.tag, c.name or c.tag))
                added += 1
            review = c.confidence is not None and c.confidence < s.conf_min
            con.execute("UPDATE tags SET x=?, y=?, z=?, confidence=?, sightings=?, read_as=?, evidence_crop=?, "
                        "needs_review=? WHERE id=?",
                        (c.x, c.y, c.z, c.confidence, c.sightings, c.read_as, c.evidence_crop, int(review), c.tag))
        con.commit()
        return {"updated": updated, "added": added}

    @app.post("/detect")
    async def detect(file: UploadFile = File(...), image_name: Optional[str] = None):
        """Stub until a trained model exists: reads the upload and returns NO detections (nothing is invented).
        When best.pt is wired in, this returns the format that POST /detections accepts."""
        data = await file.read()
        return {"model": "stub", "image": image_name or file.filename, "bytes": len(data), "detections": [],
                "note": "stub detector: no model loaded yet, returns no detections"}

    @app.post("/detections")
    def post_detections(items: List[DetectionIn]):
        """Import model output. Each relay becomes a device, attached to the nearest cabinet by position."""
        cabinets = con.execute("SELECT * FROM tags WHERE x IS NOT NULL").fetchall()
        created, ignored = [], []
        for d in items:
            if d.class_name not in DEVICE_CLASSES:
                ignored.append({"image": d.image, "class": d.class_name, "reason": "not a device class"})
                continue
            dtype, model = DEVICE_CLASSES[d.class_name]
            reasons, tag_id, p = [], None, d.position
            if d.conf < s.conf_min:
                reasons.append(f"confidence {d.conf:.2f} below {s.conf_min}")
            if p is None:
                reasons.append("no position")
            elif not cabinets:
                reasons.append("no cabinet positions loaded")
            else:
                def dist(c):
                    return math.dist((c["x"], c["y"], c["z"]), (p.x, p.y, p.z))
                best = min(cabinets, key=dist)
                if dist(best) <= s.max_assign_dist:
                    tag_id = best["id"]
                else:
                    reasons.append(f"nearest cabinet {best['id']} is {dist(best):.1f} m away "
                                   f"(limit {s.max_assign_dist} m)")
            cur = con.execute(
                "INSERT INTO devices(tag_id, class, type, model, confidence, x, y, z, ocr_text, ocr_conf, "
                "source_image, box, needs_review, review_reasons) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tag_id, d.class_name, dtype, model, d.conf, p.x if p else None, p.y if p else None,
                 p.z if p else None, json.dumps(d.ocr), d.ocr_conf, d.image,
                 json.dumps(d.box) if d.box else None, int(bool(reasons)), json.dumps(reasons)))
            created.append(device_out(con.execute("SELECT * FROM devices WHERE id = ?", (cur.lastrowid,)).fetchone()))
        con.commit()
        return {"created": created, "ignored": ignored}

    @app.get("/devices")
    def list_devices(unassigned: Optional[bool] = None):
        sql = "SELECT * FROM devices" + (" WHERE tag_id IS NULL" if unassigned else "")
        return {"devices": [device_out(r) for r in con.execute(sql + " ORDER BY id")]}

    @app.get("/documents")
    def list_documents(model: Optional[str] = None):
        return {"documents": knowledge.all_documents(kb, model)}

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: int):
        r = knowledge.get_document(kb, doc_id)
        if r is None:
            raise HTTPException(404, f"document {doc_id} not found")
        return knowledge.doc_dict(r)

    @app.get("/documents/{doc_id}/file")
    def get_document_file(doc_id: int):
        r = knowledge.get_document(kb, doc_id)
        if r is None:
            raise HTTPException(404, f"document {doc_id} not found")
        path = (s.docs_dir / r["file_path"]).resolve()
        if s.docs_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, f"PDF for document {doc_id} is not on disk ({r['file_path']})")
        return FileResponse(path, media_type="application/pdf", filename=path.name)

    @app.post("/ask")
    def ask(body: AskIn):
        """Retrieval over the manuals with page citations. 'answer' stays null: no LLM is wired in yet."""
        if kb is None:
            raise HTTPException(503, "knowledge base not found; run scripts/ingest_docs.py")
        doc_ids = None
        if body.tag_id:
            t = get_tag_row(body.tag_id)
            models = set(json.loads(t["doc_models"]))
            models |= {r["model"] for r in con.execute(
                "SELECT DISTINCT model FROM devices WHERE tag_id = ? AND model IS NOT NULL", (body.tag_id,))}
            doc_ids = [d["id"] for d in knowledge.documents_for_models(kb, models)]
            if not doc_ids:
                return {"found": False, "answer": None, "passages": [], "note": f"tag {body.tag_id} has no documents"}
        hits = knowledge.search(kb, body.question, body.k, body.model, doc_ids)
        passages = [{
            "document_id": h["document_id"], "document": h["document"], "model": h["model"],
            "section_path": h["section_path"], "page_start": h["page_start"], "page_end": h["page_end"],
            "text": h["text"][:1200], "link": f"/documents/{h['document_id']}/file#page={h['page_start']}",
        } for h in hits]
        return {"found": bool(passages), "answer": None, "passages": passages,
                "note": None if passages else "not found in the manuals"}

    return app


app = create_app()
