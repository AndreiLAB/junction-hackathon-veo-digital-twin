"""FastAPI backend: serves cabinet tags (with devices, documents, pictures) to the viewer.

Run from the repo root:  uvicorn backend.main:app --reload
"""
import json
import math
from typing import List, Optional

import csv as _csv
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, knowledge
from . import methods as M
from .config import ROOT, Settings, load_settings
from .storage import LocalStorage

# detector class -> (device type, manual model). nameplate is evidence for OCR, not a device.
DEVICE_CLASSES = {
    "abb_relion_615": ("ABB 615 protection relay", "ABB 615"),
    "relay_front": ("ABB 615 protection relay", "ABB 615"),
    "relay_rear": ("ABB 615 protection relay", "ABB 615"),
    # the breaker seen through the window marked "VD4" on the lower door of the UniGear panels: carries the VD4 manual
    "vd4_breaker_window": ("VD4 circuit breaker", "VD4"),
}
# the cabinet itself: not a device, but evidence for the cabinet tag (panel_model, source, confidence)
PANEL_CLASS, PANEL_MODEL = "unigear_zs2_panel", "UniGear ZS2"
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


class LocateIn(BaseModel):
    """Image method: a photo (pose taken from cameras.json) and/or a pose entered by hand."""
    photo: Optional[str] = Field(default=None, description="photo name, e.g. img_067.jpg")
    position: Optional[Position] = Field(default=None, description="camera position x, y, z (E57 coordinates)")
    rotation_wxyz: Optional[List[float]] = Field(default=None, min_length=4, max_length=4, description="camera orientation quaternion [w, x, y, z]")


class ExportIn(LocateIn):
    method: Literal["e57", "image"] = "e57"
    format: Optional[str] = Field(default=None, description="manual: json|csv; matterport: model_api|sdk")


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

    app = FastAPI(title="VEO360 auto-tagging backend", version="0.2.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"], allow_headers=["*"])
    cameras = M.load_cameras(s.photos_dir)
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
            "panel_model": r["panel_model"], "panel_source": r["panel_source"], "panel_confidence": r["panel_confidence"],
            "found_by_e57": bool(r["e57_found"]),
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

    def apply_cabinets(items):
        updated, added = 0, 0
        for c in items:
            if con.execute("SELECT 1 FROM tags WHERE id = ?", (c.tag,)).fetchone():
                updated += 1
            else:
                con.execute("INSERT INTO tags(id, name, created_by) VALUES (?,?,'import')", (c.tag, c.name or c.tag))
                added += 1
            review = c.confidence is not None and c.confidence < s.conf_min
            con.execute("UPDATE tags SET x=?, y=?, z=?, confidence=?, sightings=?, read_as=?, evidence_crop=?, needs_review=?, e57_found=1 WHERE id=?",
                        (c.x, c.y, c.z, c.confidence, c.sightings, c.read_as, c.evidence_crop, int(review), c.tag))
        con.commit()
        return {"updated": updated, "added": added}

    @app.post("/import/cabinets")
    def import_cabinets(items: List[CabinetIn]):
        """Load cabinet positions/confidence found by the E57 pipeline (find_labels.py tags.json, or physical_tags.csv). Marks them found_by_e57."""
        return apply_cabinets(items)

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
        created, ignored, panels = [], [], []
        for d in items:
            if d.class_name == PANEL_CLASS:        # cabinet evidence, not a device
                p, why = d.position, None
                if p is None:
                    why = "no position"
                elif not cabinets:
                    why = "no cabinet positions loaded"
                else:
                    def pdist(c):
                        return math.dist((c["x"], c["y"], c["z"]), (p.x, p.y, p.z))
                    near = min(cabinets, key=pdist)
                    if pdist(near) > s.panel_assign_dist:
                        why = f"nearest cabinet {near['id']} is {pdist(near):.2f} m away (limit {s.panel_assign_dist} m)"
                    elif d.conf < s.conf_min:
                        why = f"confidence {d.conf:.2f} below {s.conf_min}"
                if why:
                    panels.append({"image": d.image, "cabinet": None, "applied": False, "reason": why})
                else:
                    row = con.execute("SELECT doc_models, panel_confidence FROM tags WHERE id = ?", (near["id"],)).fetchone()
                    models = json.loads(row["doc_models"])
                    if PANEL_MODEL not in models:
                        models.append(PANEL_MODEL)
                    con.execute("UPDATE tags SET panel_model=?, panel_source='detected', panel_confidence=?, doc_models=? WHERE id=?",
                                (PANEL_MODEL, max(d.conf, row["panel_confidence"] or 0.0), json.dumps(models), near["id"]))
                    panels.append({"image": d.image, "cabinet": near["id"], "applied": True, "reason": "panel detected at this cabinet"})
                continue
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
        return {"created": created, "ignored": ignored, "panels": panels}

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

    # ---------------------------------------------------------------- the two methods, as the frontend uses them
    def e57_rows():
        return con.execute("SELECT * FROM tags WHERE e57_found = 1 ORDER BY rowid").fetchall()

    @app.get("/methods")
    def methods_info():
        found = len(e57_rows())
        total = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        return {
            "e57": {"available": found > 0, "cabinets_found": found, "cabinets_known": total,
                    "description": "Cabinets found by the E57 pipeline (the scan is already processed; no upload)."},
            "image": {"available": bool(cameras), "photos": len(cameras), "manual_pose": True,
                      "description": "A photo plus the pose it was taken from. Cabinets in view are found by geometry; no cabinet in view means no tag."}}

    @app.get("/methods/e57/tags")
    def e57_tags():
        """E57 method: every cabinet the E57 pipeline found, with tag, picture and manuals. `missing` = known cabinets it did not find."""
        tags = [tag_out(r) for r in e57_rows()]
        missing = [r["id"] for r in con.execute("SELECT id FROM tags WHERE e57_found = 0 ORDER BY rowid")]
        return {"method": "e57", "site": s.site, "count": len(tags), "missing": missing,
                "note": "Positions are E57 coordinates. Devices appear only when a real detection has been imported.", "tags": tags}

    def photo_list():
        return [M.photo_summary(e) for e in cameras.values()]

    @app.get("/photos")
    def list_photos(category: Optional[Literal["none", "single", "multiple"]] = None, cabinet: Optional[str] = None):
        """Image method: the available photos with their pose and which cabinets they show."""
        if not cameras:
            raise HTTPException(503, "photos are not available on this server (PHOTOS_DIR with cameras.json is missing)")
        ps = photo_list()
        if category:
            ps = [p for p in ps if p["category"] == category]
        if cabinet:
            ps = [p for p in ps if cabinet in p["cabinets"]]
        return {"count": len(ps), "photos": ps}

    @app.get("/photos/{name}/image")
    def photo_image(name: str, w: int = Query(1280, ge=128, le=4096)):
        """The photo as a JPEG no wider than w px. Boxes in other responses are in 4096-px photo coordinates: scale them by shown_width / 4096."""
        if name not in cameras or not (s.photos_dir / name).is_file():
            raise HTTPException(404, f"photo {name} not found")
        cache = s.thumbs_dir / f"{Path(name).stem}_{w}.jpg"
        if not cache.is_file():
            from PIL import Image
            im = Image.open(s.photos_dir / name).convert("RGB")
            im.thumbnail((w, w))
            cache.parent.mkdir(parents=True, exist_ok=True)
            im.save(cache, "JPEG", quality=85)
        return FileResponse(cache, media_type="image/jpeg")

    def locate_view(body: LocateIn):
        if body.photo:
            e = cameras.get(body.photo)
            if e is None:
                raise HTTPException(404, f"photo {body.photo} not found")
            pos = [body.position.x, body.position.y, body.position.z] if body.position else list(e["position"])
            rot = list(body.rotation_wxyz) if body.rotation_wxyz else list(e["rotation_wxyz"])
        elif body.position is not None and body.rotation_wxyz is not None:
            pos, rot = [body.position.x, body.position.y, body.position.z], list(body.rotation_wxyz)
        else:
            raise HTTPException(422, "give a photo name, or both position and rotation_wxyz")
        try:
            cam = M.Cam(pos, rot)
        except ValueError as err:
            raise HTTPException(422, str(err))
        vis, why = M.visible_cabinets(cam)
        cabs = []
        for v in vis:
            cid = v["cabinet"]
            row = con.execute("SELECT * FROM tags WHERE id = ?", (cid,)).fetchone()
            if row is None:
                continue
            det = []
            if body.photo:
                det = [device_out(d) for d in con.execute("SELECT * FROM devices WHERE tag_id = ? AND source_image = ? ORDER BY id", (cid, body.photo)).fetchall()]
            for d in det:
                b = d["source"]["box"]
                d["xyxy"] = [b[0], b[1], b[0] + b[2], b[1] + b[3]] if b else None
            cabs.append({"tag": tag_out(row), "view": v, "assets": M.expected_assets(cam, cid), "detected_devices": det})
        cat = "none" if not cabs else "single" if len(cabs) == 1 else "multiple"
        msg = (why or "No cabinet in view: no tag is created for this position.") if not cabs else f"{len(cabs)} cabinet{'s' if len(cabs) > 1 else ''} in view"
        return {"method": "image", "photo": body.photo,
                "image_url": f"/photos/{body.photo}/image" if body.photo and body.photo in cameras else None,
                "pose": {"position": dict(zip("xyz", pos)), "rotation_wxyz": rot}, "category": cat, "message": msg, "cabinets": cabs,
                "frame_px": M.FRAME,
                "note": "assets are EXPECTED positions from measured geometry (status 'expected'); detected_devices are real model detections once imported"}

    @app.post("/locate")
    def locate(body: LocateIn):
        """Image method: which cabinets does this photo / pose show, with their tags, pictures, manuals and where each asset must appear."""
        return locate_view(body)

    # ---------------------------------------------------------------- exports
    def export_tags(body: ExportIn):
        if body.method == "e57":
            return [tag_out(r) for r in e57_rows()], None
        loc = locate_view(body)
        return [c["tag"] for c in loc["cabinets"]], loc["message"]

    def run_manual(body: ExportIn):
        fmt = (body.format or "json").lower()
        if fmt not in ("json", "csv"):
            raise HTTPException(422, "format must be json or csv")
        tags, note = export_tags(body)
        sheet = M.manual_sheet(tags, body.method, s.public_base_url, note)
        if fmt == "csv":
            return Response(M.manual_csv(sheet), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="manual_tagging_{body.method}.csv"'})
        return sheet

    def run_matterport(body: ExportIn):
        fmt = (body.format or "model_api").lower()
        if fmt not in ("model_api", "sdk"):
            raise HTTPException(422, "format must be model_api or sdk")
        tags, note = export_tags(body)
        out = M.matterport_export(tags, body.method, s.public_base_url, fmt)
        out["message"] = note
        return out

    @app.post("/export/manual")
    def export_manual(body: ExportIn):
        """A manual tagging sheet for the selected method: what a person must create by hand in the digital twin (JSON, or CSV with format=csv)."""
        return run_manual(body)

    @app.get("/export/manual")
    def export_manual_get(method: Literal["e57", "image"] = "e57", photo: Optional[str] = None, format: Optional[str] = None):
        return run_manual(ExportIn(method=method, photo=photo, format=format))

    @app.post("/export/matterport")
    def export_matterport(body: ExportIn):
        """Tags in Matterport's shape (format=model_api: addMattertag input + GraphQL mutation; format=sdk: Tag.add descriptors). Dry run: nothing is sent."""
        return run_matterport(body)

    @app.get("/export/matterport")
    def export_matterport_get(method: Literal["e57", "image"] = "e57", photo: Optional[str] = None, format: Optional[str] = None):
        return run_matterport(ExportIn(method=method, photo=photo, format=format))

    # what the E57 pipeline already found is loaded once at start (no upload needed)
    if s.e57_results and Path(s.e57_results).is_file() and not e57_rows():
        best = {}
        for r in _csv.DictReader(open(s.e57_results, encoding="utf-8")):
            if r.get("type") == "cubicle" and (r["label"] not in best or float(r["confidence"]) > float(best[r["label"]]["confidence"])):
                best[r["label"]] = r
        apply_cabinets([CabinetIn(tag=k, x=float(v["x"]), y=float(v["y"]), z=float(v["z"]), confidence=float(v["confidence"]),
                                  sightings=int(float(v["observation_count"]))) for k, v in best.items()])

    return app


app = create_app()
