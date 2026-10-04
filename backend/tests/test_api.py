"""API tests. Tags/devices use a temp database; the manuals come from the committed data/knowledge.db and docs/."""
import pytest
from fastapi.testclient import TestClient

from backend.config import ROOT, Settings
from backend.main import create_app


@pytest.fixture
def client(tmp_path):
    s = Settings(veo_db=tmp_path / "veo.db", knowledge_db=ROOT / "data" / "knowledge.db",
                 docs_dir=ROOT / "docs", assets_dir=tmp_path / "assets", site="test",
                 conf_min=0.5, max_assign_dist=2.0)
    return TestClient(create_app(s))


CABINETS = [{"tag": "H05", "x": 1.0, "y": 1.0, "z": 1.0, "confidence": 0.9},
            {"tag": "H04", "x": 5.0, "y": 1.0, "z": 1.0, "confidence": 0.9}]


def detection(**kw):
    d = {"image": "scan_012_skybox_3", "class": "abb_relion_615", "conf": 0.9, "box": [10, 20, 30, 40],
         "ocr": ["615"], "ocr_conf": 0.8, "position": {"x": 1.1, "y": 1.0, "z": 1.2}}
    d.update(kw)
    return d


def test_health_and_ten_seeded_cabinets(client):
    h = client.get("/health").json()
    assert h["tags"] == 10 and h["knowledge_base"] and h["documents"] == 3
    body = client.get("/tags").json()
    assert body["count"] == 10 and body["site"] == "test"
    assert [t["id"] for t in body["tags"]][:2] == ["H05", "H04"]
    assert all(t["position"] is None and t["devices"] == [] for t in body["tags"])   # nothing invented


def test_h05_has_cabinet_manuals_others_none(client):
    h05 = client.get("/tags/H05").json()
    assert sorted(d["model"] for d in h05["documents"]) == ["UniGear ZS2", "VD4"]
    assert client.get("/tags/H04").json()["documents"] == []
    assert client.get("/tags/NOPE").status_code == 404


def test_document_file_is_a_pdf(client):
    doc = client.get("/tags/H05").json()["documents"][0]
    r = client.get(doc["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"
    assert client.get("/documents/9999").status_code == 404


def test_import_cabinets_sets_positions(client):
    assert client.post("/import/cabinets", json=CABINETS).json() == {"updated": 2, "added": 0}
    assert client.get("/tags/H05").json()["position"] == {"x": 1.0, "y": 1.0, "z": 1.0}


def test_detection_attaches_to_nearest_cabinet_with_relay_manual(client):
    client.post("/import/cabinets", json=CABINETS)
    r = client.post("/detections", json=[detection()]).json()
    dev = r["created"][0]
    assert dev["tag_id"] == "H05" and dev["needs_review"] is False
    assert dev["documents"][0]["model"] == "ABB 615" and dev["documents"][0]["pages"] == 1985
    assert client.get("/tags/H05").json()["devices"][0]["id"] == dev["id"]


def test_low_confidence_far_or_positionless_detections_need_review(client):
    client.post("/import/cabinets", json=CABINETS)
    out = client.post("/detections", json=[
        detection(conf=0.2),
        detection(position={"x": 50, "y": 50, "z": 50}),
        detection(position=None),
        detection(**{"class": "nameplate"}),
    ]).json()
    low, far, nopos = out["created"]
    assert low["needs_review"] and "below" in low["review_reasons"][0] and low["tag_id"] == "H05"
    assert far["needs_review"] and far["tag_id"] is None
    assert nopos["needs_review"] and nopos["review_reasons"] == ["no position"]
    assert out["ignored"][0]["class"] == "nameplate"
    assert len(client.get("/devices", params={"unassigned": True}).json()["devices"]) == 2


def test_detect_stub_returns_no_detections(client):
    r = client.post("/detect", files={"file": ("a.jpg", b"123", "image/jpeg")}).json()
    assert r["model"] == "stub" and r["detections"] == []


def test_image_upload_and_placeholder_flag(client):
    assert client.get("/tags/H05").json()["image"] is None
    r = client.put("/tags/H05/image", files={"file": ("x.jpg", b"jpgbytes", "image/jpeg")}).json()
    assert r["image"] == "/assets/context/H05.jpg" and r["image_is_placeholder"] is True
    assert client.get(r["image"]).content == b"jpgbytes"
    assert client.put("/tags/H05/image", files={"file": ("x.gif", b"1", "image/gif")}).status_code == 422


def test_patch_tag(client):
    r = client.patch("/tags/H05", json={"needs_review": True, "x": 2.0, "y": 3.0, "z": 4.0}).json()
    assert r["needs_review"] and r["position"] == {"x": 2.0, "y": 3.0, "z": 4.0}
    assert len(client.get("/tags", params={"needs_review": True}).json()["tags"]) == 1


def test_ask_cites_pages_and_says_not_found(client):
    r = client.post("/ask", json={"question": "What do the READY START TRIP LEDs show?", "model": "ABB 615"}).json()
    assert r["found"] and r["answer"] is None and r["passages"][0]["page_start"] == 55
    assert r["passages"][0]["link"].endswith("#page=55")
    nf = client.post("/ask", json={"question": "What is the capital of Finland?"}).json()
    assert nf["found"] is False
    scoped = client.post("/ask", json={"question": "closing spring charged", "tag_id": "H05"}).json()
    assert scoped["found"] and {p["model"] for p in scoped["passages"]} <= {"UniGear ZS2", "VD4"}
    assert client.post("/ask", json={"question": "anything", "tag_id": "H04"}).json()["found"] is False


def test_vd4_window_becomes_a_device_with_the_vd4_manual(client):
    client.post("/import/cabinets", json=CABINETS)
    det = detection(**{"class": "vd4_breaker_window", "position": {"x": 1.0, "y": 1.1, "z": 0.8}})
    dev = client.post("/detections", json=[det]).json()["created"][0]
    assert dev["tag_id"] == "H05" and dev["type"] == "VD4 circuit breaker" and dev["model"] == "VD4"
    assert dev["documents"][0]["model"] == "VD4" and dev["documents"][0]["pages"] == 132 and not dev["needs_review"]
    # the same cabinet carries the relay and the breaker as separate devices
    client.post("/detections", json=[detection()])
    assert sorted(d["class"] for d in client.get("/tags/H05").json()["devices"]) == ["abb_relion_615", "vd4_breaker_window"]


def test_vd4_without_position_is_flagged_and_other_classes_still_ignored(client):
    out = client.post("/detections", json=[detection(**{"class": "vd4_breaker_window", "position": None}),
                                           detection(**{"class": "other_hmi"}),
                                           detection(**{"class": "unigear_zs2_panel"})]).json()
    assert out["created"][0]["needs_review"] and out["created"][0]["review_reasons"] == ["no position"]
    assert [i["class"] for i in out["ignored"]] == ["other_hmi", "unigear_zs2_panel"]
