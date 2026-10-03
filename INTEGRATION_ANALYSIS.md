# Code-Level Integration Analysis

Based on an audit of the `backend` branch and the recently merged branches, here are the technical answers to your integration questions:

### A. E57 Extraction
**Q:** Given one E57, what exact function/script produces the extracted images?
**A:** There are currently two redundant scripts:
1. `extract_e57.py`: Uses `pye57` and SQLite to properly parse images, calculate yaw/pitch/roll via quaternions, and populate the `veo_localization.db`.
2. `synthetic_pipeline/extract_images.py`: A legacy standalone script that iterates `e57.root["images2D"]` and dumps JPEGs alongside a simple `cameras.json`.

### B. Single Image Detection
**Q:** Given one image, what exact code detects the cabinet label, ABB equipment, OCR text, and bounding box?
**A:** `synthetic_pipeline/find_labels.py` is the only active vision script.
- **Cabinet label & Bounding box**: Detected using `cv2.inRange` for orange tape HSV masking in the `orange_tapes(img)` function.
- **OCR text**: Extracts text using `RapidOCR` inside the `labels_in_photo(ocr, img)` function. It uses `difflib.SequenceMatcher` in `match_tag(text)` to map raw OCR text (e.g., "SOLAR 1") to known cabinet IDs (e.g., "H04").
- **ABB Equipment**: **Not implemented yet.** There is no active computer vision code to detect ABB equipment.

### C. Current ABB Recognition State
**Q:** Is ABB recognition currently OCR only, template matching, embedding, or an ML classifier?
**A:** It is a **STUB (Planned ML Classifier / Object Detection)**. 
Evidence: In `backend/main.py`, the `/detect` endpoint explicitly states: `"""Stub until a trained model exists: reads the upload and returns NO detections."""` It expects a future trained model (likely YOLO/object detection) that outputs bounding boxes and classes like `"abb_relion_615"`.

### D. Recognition Output Data Structure
**Q:** Show the exact output object/data structure produced by recognition.
**A:** The future ABB detector is expected to produce the `DetectionIn` Pydantic schema defined in `backend/main.py`:
```json
{
  "image": "scan_012_skybox_3.jpg",
  "class": "abb_relion_615",
  "conf": 0.95,
  "box": [102.0, 50.5, 200.0, 150.0],
  "ocr": ["REX615"],
  "ocr_conf": 0.88,
  "position": {"x": 1.5, "y": -2.0, "z": 0.8}
}
```

### E & F. 2D to 3D Conversion (Raycasting)
**Q:** How is the 2D detection converted into a 3D coordinate? Does it actually intersect the point cloud?
**A:** **Yes, it physically intersects the point cloud.**
In `synthetic_pipeline/find_labels.py`, the following sequence occurs:
1. It loads the massive raw XYZ point cloud using `e57.read_scan(i)`.
2. `project()` applies camera extrinsics (`Rc2w`, `C`) and intrinsics (`f`, `cx`, `cy`) to project every single 3D point into 2D camera pixels `(u, v)`.
3. It filters down to points that land precisely inside the 2D bounding box: `near = ok & (np.abs(u - h["u"]) < rad) & (np.abs(v - h["v"]) < rad)`.
4. It isolates the front-most surface: `front = z < z.min() * 1.05 + 0.03`.
5. The final 3D anchor is the median of those points: `np.median(xyz[near][front], axis=0)`.

### G. Knowledge Base Mapping
**Q:** How does the knowledge base map "REX615" → document → page/section?
**A:** 
1. The mapping is hardcoded via a dictionary in `backend/main.py` (`DEVICE_CLASSES = {"abb_relion_615": ("ABB 615 protection relay", "ABB 615")}`).
2. When the backend receives an "abb_relion_615" device, it associates it with the string `"ABB 615"`.
3. `scripts/kb_search.py` and `backend/knowledge.py` query the FTS5 SQLite index (`data/knowledge.db`) filtering by `AND d.model = 'ABB 615'`.
4. The database returns the matching `document_id`, `page_start`, and `section_path` (extracted from the PDF structure).

### H. Existing Tag Generators
**Q:** What existing code already generates tags.csv/tags.json?
**A:** `synthetic_pipeline/find_labels.py` generates `tags.json` (line 299) and `tags.csv` (line 301) by aggregating and averaging the 3D positions of all label sightings.

### I. Duplication & Technical Debt
**Q:** Which files duplicate functionality and should eventually be merged?
- `synthetic_pipeline/extract_images.py` heavily duplicates the cleaner, DB-driven `extract_e57.py`.
- `synthetic_pipeline/inspect_e57.py` duplicates `inspect_dataset.py`.
- `synthetic_pipeline/find_labels.py` is a monolithic God-script (it does E57 parsing, CV, DB generation, and raycasting in one file).

### J. The Minimum Integration Goal
**Q:** What is the MINIMUM code we need to turn everything into `process_e57.py`?
**A:** We need to orchestrate the existing components into a linear flow:
1. `extract_e57.py`: (Already works) Parses E57 to DB + images.
2. `vision/ocr.py`: (Needs isolating) Strip the `RapidOCR` and HSV masking logic out of `find_labels.py` so it just accepts an image and returns 2D text boxes.
3. `vision/detector.py`: (New) A mock detector returning hardcoded `abb_relion_615` bounding boxes for demo purposes.
4. `e57/raycast.py`: (Needs isolating) Strip the `project()` and point-cloud intersection math out of `find_labels.py`. It should accept a 2D BBox + Image DB ID, and return a 3D `(x, y, z)` coordinate.
5. `export_tags.py`: (New) Takes the 3D-mapped labels and assets and formats them into `outputs/tags.json` matching the `CabinetIn` and `DetectionIn` schemas expected by `backend/main.py`.
