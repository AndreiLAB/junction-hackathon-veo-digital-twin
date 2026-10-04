# junction-hackathon-veo-digital-twin

Digital twin platform for VEO company - Junction Hackathon Team JSC

## VEO Visual Localization Pipeline

This repository contains the complete research and prototype visual localization pipeline built for ASTM E57 digital twins (Matterport / VEO360 scans).

Given a 2D query image, the pipeline predicts:
- Most likely scan sweep / viewpoint
- 3D Camera Position $(X, Y, Z)$
- Viewing Direction $(\text{Yaw}, \text{Pitch}, \text{Roll})$ and forward camera vector
- Top-K nearest visual candidates with cosine similarity scores

---

### Key Capabilities

1. **Streaming E57 Node Extraction (`extract_e57.py`)**:
   - Parses `Data3D` scans and `Images2D` nodes without loading multi-gigabyte point clouds into memory.
   - Decodes original embedded skybox images (JPEG/PNG).
   - Extracts camera translation and quaternion rotation $(q_w, q_x, q_y, q_z)$.
   - Computes derived 3D rotation matrices, forward vectors, and spherical orientation (Yaw/Pitch/Roll).

2. **Metadata & Relational Storage (`db/database.py`)**:
   - Structured SQLite storage (`dataset/veo_localization.db`) with unique GUID constraints for scans, images, and feature vectors.

3. **Direction Calibration (`calibrate_direction.py`)**:
   - Generates contact sheets of skybox faces $(0\text{--}5)$ alongside computed pose coordinates and forward vectors to verify camera coordinate axes.

4. **Deep Vision Embeddings (`build_embeddings.py`)**:
   - Extracts semantic visual features using pretrained **DINOv2** (`dinov2_vits14`), $L_2$-normalized for cosine similarity retrieval.

5. **Multi-Task & Leave-One-Scan-Out Modeling (`train_model.py`)**:
   - Lightweight neural head predicting scan locations and discrete direction sectors.
   - Evaluates generalization across unseen scans.

6. **Rigorous Retrieval Evaluation (`evaluate.py`)**:
   - Tests visual localization using dynamically generated aggressive augmentations (Color jitter, random perspective, random resized crop, Gaussian blur).
   - Generates retrieval contact sheets (`QUERY | TOP 1 | TOP 2 | TOP 3 | TOP 4 | TOP 5`) and comprehensive metrics.

7. **Inference & Visualization (`localize.py`, `visualize_localization.py`)**:
   - CLI tool for querying any input image against the indexed digital twin and plotting visual localization sheets.

---

### Project Structure

```
├── config.py                     # Central configuration (paths, axes, models)
├── extract_e57.py                # E57 image, pose, and metadata extractor
├── inspect_dataset.py            # Quality check & HTML dataset visualizer
├── calibrate_direction.py        # Direction calibration & skybox contact sheet
├── build_embeddings.py           # DINOv2 feature extraction pipeline
├── train_model.py                # Multi-task orientation & scan classifier
├── evaluate.py                   # Rigorous retrieval benchmark suite
├── localize.py                   # Single-image query visual localization
├── visualize_localization.py     # Visual query debugger
├── pipeline.py                   # End-to-end pipeline runner
├── requirements.txt              # Dependencies
├── db/
│   └── database.py               # SQLite schema & array adapters
├── e57/
│   └── geometry.py               # Quaternion & 3D vector transformations
├── vision/                       # Vision modules
├── outputs/                      # Evaluation reports, models, & visualizations
└── dataset/                      # Extracted images & SQLite database
```

---

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AndreiLAB/junction-hackathon-veo-digital-twin.git
   cd junction-hackathon-veo-digital-twin
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

#### 1. Full Pipeline Execution
To run the complete extraction, calibration, embedding generation, training, and evaluation:
```bash
python pipeline.py --e57 path/to/cloud.e57
```

#### 2. Query Localization
To localize an arbitrary photo against the digital twin:
```bash
python localize.py --image path/to/query.jpg
```

To render an annotated visual comparison contact sheet:
```bash
python visualize_localization.py --image path/to/query.jpg
```

#### 3. Rigorous Evaluation
To evaluate localization robustness against synthetic perturbations:
```bash
python evaluate.py
```
Outputs:
- `outputs/evaluation_report.md`
- `outputs/evaluation_results.json`
- `outputs/retrieval_contact_sheet.jpg`


---

## Manuals knowledge base (REX615, UniGear ZS2, VD4)

Manuals converted to section-based Markdown and a searchable SQLite database, so tags can return their PDFs and an optional `/ask` can answer with page citations.

- `data/knowledge.db`: `documents` (title, model, doc number, PDF path, pages), `chunks` (section path, page range, text) and an FTS5 index `chunks_fts`.
- `docs_md/<folder>/<manual>.md`: readable Markdown with `<!-- p.N -->` page markers.
- The source PDFs are in `docs/abb_615/`, `docs/unigear_zs2/`, `docs/vd4/` (about 137 MB; the folder name sets the model).

```bash
pip install pymupdf
python scripts/ingest_docs.py                                   # rebuild (about 5 min for the 1,985-page REX615 manual)
python scripts/kb_search.py "READY LED" --model "ABB 615"       # search with citations, prints "not found" if nothing matches
python scripts/kb_eval.py                                       # retrieval test: 11/11 in top 3
```
Library use: `from kb_search import search; search(con, question, k=5, model="ABB 615")` returns chunks with document, section path and page range.


---

## Backend API (FastAPI)

Serves one **tag per cabinet** (H05, H04, H03, H02, H01, VLK, OT1, TSK1, TSK2, OKK1) with its devices (e.g. the ABB 615 relay), documents (PDFs from the knowledge base) and picture. The existing viewer (Matterport / VEO360) consumes `GET /tags`; write-back to Matterport/VEO360 is **not implemented** (the API and its write permission are unconfirmed).


### Frontend API: the two methods and the exports (version 0.2)

The frontend (`frontend/LOVABLE_PROMPT.md`, built in Lovable) uses two methods. CORS is enabled (`CORS_ORIGINS`, default `*`).

| Method and path | Parameters | Returns |
|---|---|---|
| `GET /methods` | none | what each method has: `e57.cabinets_found/known`, `image.photos` |
| `GET /methods/e57/tags` | none | **E57 method**: the cabinets the E57 pipeline already found (no upload; `outputs/physical_tags.csv` is loaded at start), each with tag, picture, manuals; `missing` lists known cabinets it did not find (TSK1) |
| `GET /photos` | query `category` (none/single/multiple), `cabinet` | **Image method**: the real photos with pose, category and the cabinets they show (503 if `PHOTOS_DIR` is missing) |
| `GET /photos/{name}/image` | query `w` (128-4096, default 1280) | the photo as a JPEG (boxes elsewhere are in 4096-px photo coordinates) |
| `POST /locate` | JSON `{photo?, position?{x,y,z}, rotation_wxyz?[w,x,y,z]}` | cabinets in view with their tags, pictures, manuals, **expected** asset boxes (panel, relay, VD4 window, look-alike display) and any detected devices of that photo. **No cabinet in view -> `cabinets: []` and no tag** |
| `POST /export/manual` (also `GET`) | JSON `{method: e57|image, photo?, position?, rotation_wxyz?, format?: json|csv}` | a manual tagging sheet: what a person creates by hand in the digital twin (name, position, picture, documents); CSV is a file download |
| `POST /export/matterport` (also `GET`) | same, `format: model_api|sdk` | Matterport-shaped tags: `model_api` = `addMattertag` input + the GraphQL mutation (variables per tag); `sdk` = `Tag.add` descriptors (session only). **Dry run**: nothing is sent; positions are E57 coordinates (not transformed); `<MATTERPORT_MODEL_ID>` / `<FLOOR_ID>` are placeholders; cabinets without a position are listed under `skipped`; unconfirmed fields are listed under `notes` |

New settings: `PHOTOS_DIR` (the VEO photos + `cameras.json`, default `VEO Images`, not in git), `THUMBS_DIR`, `PUBLIC_BASE_URL` (prefix for the picture/PDF links inside exports, default `http://localhost:8000`), `CORS_ORIGINS`, `E57_RESULTS`.
Geometry for the image method is `backend/asset_geometry.json` (hand-measured, +-0.02 m, a stable copy of `synthetic_geo/relay_geometry.json`). Expected asset boxes are geometry predictions, not detections.
To let a browser app on another machine reach the backend: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`, and expose it (for example a tunnel) and set `PUBLIC_BASE_URL` to that address.

### How the model and the backend fit together

```
skybox images --> model (detect + OCR, offline) --> detections.json --> POST /detections --> tags --> GET /tags --> viewer
```

- The **model side** reads the skybox images. The **backend never receives or needs them**; it only receives the detection results.
- Images the backend does accept are optional: the cabinet picture (`PUT /tags/{id}/image`, display only) and the `/detect` stub (a hook for running the model live later).
- A detection should carry its `position` (x, y, z in scan coordinates). Without it the device is created but left unassigned with `needs_review`. Computing it from pixel + camera pose (`locate()`) is not built yet.

### Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload          # from the repo root; docs at http://localhost:8000/docs
python -m pytest backend/tests -q          # 10 tests
```

On first start it creates `data/veo.db` and seeds the 10 cabinets (names only: no positions, no confidence, no devices). Nothing is invented; unknown values are `null`.

| Env variable | Default | Meaning |
|---|---|---|
| `VEO_DB` | `data/veo.db` | tags + devices (created automatically, git-ignored) |
| `KNOWLEDGE_DB` | `data/knowledge.db` | manuals database (see "Manuals knowledge base") |
| `DOCS_DIR` | `docs` | the PDFs |
| `ASSETS_DIR` | `assets` | images, served under `/assets` (git-ignored, share via Drive) |
| `VEO_SITE` | `eHouse` | site name in `GET /tags` |
| `CONF_MIN` | `0.5` | detections below this get `needs_review` |
| `PANEL_ASSIGN_DIST` | `0.6` | metres: a detected UniGear panel is matched to the nearest cabinet within this distance |
| `MAX_ASSIGN_DIST` | `2.0` | metres; a device farther than this from every cabinet stays unassigned (**unverified default**, tune it) |

### Endpoints

| Method and path | Parameters | Returns |
|---|---|---|
| `GET /health` | none | status, tag/device counts, whether the knowledge base loaded |
| `GET /tags` | query `needs_review` (bool), `folder` (str) | `{site, count, tags[]}`; each tag has `id, name, folder, position, confidence, needs_review, image, image_is_placeholder, read_as, sightings, evidence_crop, documents[], devices[]` |
| `GET /tags/{id}` | path `id` (e.g. `H05`) | one tag, same shape; 404 if unknown |
| `PATCH /tags/{id}` | JSON `name, x, y, z, needs_review, doc_models` (all optional) | the updated tag |
| `PUT /tags/{id}/image` | multipart `file` (.jpg/.jpeg/.png/.webp), query `placeholder` (bool, default true) | the updated tag; saved as `assets/context/{id}.<ext>` |
| `POST /import/cabinets` | JSON list of `{tag, name?, x?, y?, z?, confidence?, sightings?, read_as?, evidence_crop?}` (the format of `synthetic_pipeline/labels_out/tags.json`) | `{updated, added}`; sets cabinet positions |
| `POST /detect` | multipart `file`, query `image_name` (optional) | **stub**: `{model: "stub", detections: []}`. No model is loaded yet and nothing is faked |
| `POST /detections` | JSON list of `{image, class, conf, box?, ocr?, ocr_conf?, position?{x,y,z}}` (model output) | `{created[], ignored[]}`; relays become devices attached to the nearest cabinet |
| `GET /devices` | query `unassigned` (bool) | `{devices[]}` |
| `GET /documents` | query `model` (str, e.g. `ABB 615`) | `{documents[]}` with `id, title, model, doc_type, doc_number, pages, url` |
| `GET /documents/{id}` | path `id` (int) | one document |
| `GET /documents/{id}/file` | path `id` (int) | the PDF (`application/pdf`); open at a page with `#page=N` |
| `POST /ask` | JSON `{question, tag_id?, model?, k?}` (`k` 1 to 20, default 5) | `{found, answer, passages[], note}`; passages carry `document, section_path, page_start, page_end, text, link`. `answer` is always `null` (no LLM wired in); `found=false` means "not in the manuals" |
| `GET /assets/...` | static files | images |

**Detection input** (`POST /detections`). Device classes: `abb_relion_615` (or `relay_front` / `relay_rear`) -> "ABB 615 protection relay" + the REX615 manual; `vd4_breaker_window` -> "VD4 circuit breaker" + the VD4 manual (the breaker seen through the "VD4" window of a UniGear panel). `unigear_zs2_panel` is **not a device**: it is evidence on the cabinet tag. The nearest cabinet within `PANEL_ASSIGN_DIST` (0.6 m) of the detection's `position` gets `panel_model = "UniGear ZS2"`, `panel_source = "detected"` and `panel_confidence`; weak (< `CONF_MIN`), far or unplaced panel detections are not applied and are listed under `panels` with the reason. Any other class (`nameplate`, `other_hmi`) is ignored and reported under `ignored`. Response: `{created, ignored, panels}`. Cabinet tags carry `panel_model`, `panel_source` (`assumed` for the seeded H01-H05 until detected) and `panel_confidence`; H01-H05 are seeded with the UniGear ZS2 datasheet (H05 also with VD4, as in VEO's own tag). `box` is `[x, y, w, h]` in pixels of the **original** image. A detection is flagged `needs_review` (with `review_reasons`) if `conf < CONF_MIN`, if it has no `position`, or if no cabinet is within `MAX_ASSIGN_DIST`. Positions come from the model side (pixel + camera pose); this backend does not compute them yet.

**Documents** are attached by manual model: a relay device gets the ABB 615 manual; a cabinet gets the models listed in its `doc_models`. Only **H05** is pre-linked (UniGear ZS2 + VD4), because that is all VEO's manual H05 tag showed; the other cabinets have none until set with `PATCH /tags/{id}`.

**Storage** is behind `backend/storage.py` (`LocalStorage`: `save`, `exists`, `url`, `find`). A cloud class with the same methods can replace it later without changing the API.

### Not done yet
- Real detector behind `/detect` (waiting on the trained model).
- `locate()`: pixel + pose to x, y, z (needs the skybox images and a checked camera convention).
- Cabinet positions: load Pragati's `tags.json` with `POST /import/cabinets`.
- Matterport / VEO360 write-back adapters.
- LLM answer generation for `/ask` (retrieval and citations work).
