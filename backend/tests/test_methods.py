"""Tests for the two methods (image + pose, E57 results) and the exports.
Self-contained: the real poses of two photos are embedded and the JPEG is generated, so no VEO photos are needed."""
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import ROOT, Settings
from backend.main import create_app

POSE_067 = {"file": "img_067.jpg", "rotation_wxyz": [0.5205608606338501, 0.502113401889801, 0.48949265480041504, 0.487129807472229],
            "position": [-5.020389556884766, -4.4829206466674805, 1.4427616596221924]}     # faces H02 from 0.72 m
POSE_001 = {"file": "img_001.jpg", "rotation_wxyz": [0.49108925461769104, 0.511879026889801, 0.50017911195755, 0.49662068486213684],
            "position": [-0.01515436451882124, -0.0081004174426198, 0.866147518157959]}   # behind a steel door, outside the corridor


@pytest.fixture
def client(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    cams = [dict(e, type="pinholeRepresentation", width=4096, height=4096, focal_px_x=2048.0, focal_px_y=2048.0, cx=2048.0, cy=2048.0)
            for e in (POSE_067, POSE_001)]
    (photos / "cameras.json").write_text(json.dumps(cams))
    Image.new("RGB", (400, 400), (200, 200, 200)).save(photos / "img_067.jpg")
    s = Settings(veo_db=tmp_path / "veo.db", knowledge_db=ROOT / "data" / "knowledge.db", docs_dir=ROOT / "docs", assets_dir=tmp_path / "assets",
                 site="test", conf_min=0.5, max_assign_dist=2.0, photos_dir=photos, thumbs_dir=tmp_path / "thumbs",
                 public_base_url="http://api.test", e57_results=ROOT / "outputs" / "physical_tags.csv")
    return TestClient(create_app(s))


def test_cors_and_methods(client):
    r = client.options("/tags", headers={"Origin": "https://x.lovable.app", "Access-Control-Request-Method": "GET"})
    assert r.headers["access-control-allow-origin"] in ("*", "https://x.lovable.app")
    m = client.get("/methods").json()
    assert m["e57"]["available"] and m["e57"]["cabinets_found"] == 9
    assert m["image"]["available"] and m["image"]["photos"] == 2 and m["image"]["manual_pose"] is True


def test_e57_method_lists_what_the_pipeline_found_and_what_it_missed(client):
    r = client.get("/methods/e57/tags").json()
    assert r["method"] == "e57" and r["count"] == 9 and r["missing"] == ["TSK1"]       # TSK1's label was not read by the E57 pipeline
    ids = [t["id"] for t in r["tags"]]
    assert "H05" in ids and "TSK1" not in ids
    h05 = next(t for t in r["tags"] if t["id"] == "H05")
    assert h05["position"] and h05["found_by_e57"] and h05["devices"] == [] and h05["panel_source"] == "assumed"   # no fake devices


def test_photos_list_and_thumbnail(client):
    ph = client.get("/photos").json()
    assert ph["count"] == 2
    p67 = next(p for p in ph["photos"] if p["name"] == "img_067.jpg")
    assert p67["category"] == "single" and p67["cabinets"] == ["H02"] and p67["scan"] == 12
    assert next(p for p in ph["photos"] if p["name"] == "img_001.jpg")["category"] == "none"
    assert [p["name"] for p in client.get("/photos", params={"category": "none"}).json()["photos"]] == ["img_001.jpg"]
    img = client.get("/photos/img_067.jpg/image", params={"w": 200})
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/photos/nope.jpg/image").status_code == 404


def test_locate_photo_gives_tag_assets_and_documents(client):
    r = client.post("/locate", json={"photo": "img_067.jpg"}).json()
    assert r["category"] == "single" and r["image_url"] == "/photos/img_067.jpg/image" and r["frame_px"] == 4096
    c = r["cabinets"][0]
    assert c["tag"]["id"] == "H02" and c["view"]["depth_m"] == pytest.approx(0.724, abs=0.01)
    assert [d["model"] for d in c["tag"]["documents"]] == ["UniGear ZS2"]
    kinds = {a["class"]: a for a in c["assets"]}
    assert set(kinds) == {"unigear_zs2_panel", "abb_relion_615", "vd4_breaker_window"}
    rx0, ry0, rx1, ry1 = kinds["abb_relion_615"]["xyxy"]
    assert 500 < rx0 < 1300 and 300 < ry0 < 900 and 1.2 < (rx1 - rx0) / (ry1 - ry0) < 1.7   # where it was measured (img_067: ~x 650-1470, y 550-1130)
    assert all(a["status"] == "expected" for a in c["assets"]) and c["detected_devices"] == []


def test_locate_manual_pose_and_the_no_cabinet_rule(client):
    same = client.post("/locate", json={"position": {"x": POSE_067["position"][0], "y": POSE_067["position"][1], "z": POSE_067["position"][2]},
                                        "rotation_wxyz": POSE_067["rotation_wxyz"]}).json()
    assert same["photo"] is None and same["cabinets"][0]["tag"]["id"] == "H02"
    none = client.post("/locate", json={"photo": "img_001.jpg"}).json()                      # camera behind the door
    assert none["category"] == "none" and none["cabinets"] == [] and "corridor" in none["message"]
    open_space = client.post("/locate", json={"position": {"x": -4.0, "y": -3.0, "z": 1.4}, "rotation_wxyz": [1, 0, 0, 0]}).json()
    assert open_space["cabinets"] == [] and "no tag" in open_space["message"].lower() or open_space["category"] != "none"
    assert client.post("/locate", json={}).status_code == 422
    assert client.post("/locate", json={"position": {"x": 0, "y": 0, "z": 0}, "rotation_wxyz": [0, 0, 0, 0]}).status_code == 422
    assert client.post("/locate", json={"photo": "missing.jpg"}).status_code == 404


def test_locate_shows_detected_devices_of_that_photo(client):
    det = {"image": "img_067.jpg", "class": "abb_relion_615", "conf": 0.9, "box": [650, 550, 820, 570], "position": {"x": -5.676, "y": -4.78, "z": 1.84}}
    client.post("/detections", json=[det])
    c = client.post("/locate", json={"photo": "img_067.jpg"}).json()["cabinets"][0]
    assert len(c["detected_devices"]) == 1 and c["detected_devices"][0]["xyxy"] == [650, 550, 1470, 1120]


def test_manual_sheet_for_both_methods(client):
    e = client.post("/export/manual", json={"method": "e57"}).json()
    assert e["format"] == "manual-tagging-sheet" and e["count"] == 9 and e["rows"][0]["position_e57"]
    h05 = next(r for r in e["rows"] if r["tag_name"].startswith("H05"))
    assert h05["panel_model"] == "UniGear ZS2" and any(d["url"].startswith("http://api.test/documents/") for d in h05["documents"])
    i = client.post("/export/manual", json={"method": "image", "photo": "img_067.jpg"}).json()
    assert i["count"] == 1 and i["rows"][0]["tag_name"].startswith("H02")
    nothing = client.post("/export/manual", json={"method": "image", "photo": "img_001.jpg"}).json()
    assert nothing["count"] == 0 and nothing["note"]
    csv_r = client.get("/export/manual", params={"method": "image", "photo": "img_067.jpg", "format": "csv"})
    assert csv_r.headers["content-type"].startswith("text/csv") and "attachment" in csv_r.headers["content-disposition"] and "H02" in csv_r.text
    assert client.post("/export/manual", json={"method": "e57", "format": "xml"}).status_code == 422


def test_matterport_export_shapes(client):
    m = client.post("/export/matterport", json={"method": "e57"}).json()
    assert m["format"] == "matterport-model-api" and m["count"] == 9 and m["skipped"] == [] and m["modelId"] == "<MATTERPORT_MODEL_ID>"
    t = next(t for t in m["tags"] if t["label"].startswith("H05"))
    assert set(t) >= {"label", "description", "enabled", "anchorPosition", "stemEnabled", "stemNormal", "stemLength", "mediaUrl"}
    assert set(t["anchorPosition"]) == {"x", "y", "z"} and t["mediaUrl"].startswith("http://api.test/documents/") and "UniGear ZS2" in t["description"]
    assert "addMattertag" in m["graphql"]["mutation"] and len(m["graphql"]["variables"]) == 9
    assert m["graphql"]["variables"][0]["tag"]["floorId"] == "<FLOOR_ID>"
    sdk = client.get("/export/matterport", params={"method": "image", "photo": "img_067.jpg", "format": "sdk"}).json()
    assert sdk["format"] == "matterport-showcase-sdk" and sdk["count"] == 1 and set(sdk["tags"][0]) >= {"label", "anchorPosition", "stemVector", "attachmentUrls"}
    assert any("session" in n for n in sdk["notes"])
    assert client.post("/export/matterport", json={"method": "e57", "format": "x"}).status_code == 422


def test_a_cabinet_without_a_position_is_skipped_in_matterport(client):
    client.post("/import/cabinets", json=[{"tag": "NEW1", "name": "NEW1"}])      # known cabinet, no position
    r = client.post("/export/matterport", json={"method": "e57"}).json()
    assert {"tag": "NEW1", "reason": "no position yet"} in r["skipped"] and r["count"] == 9
