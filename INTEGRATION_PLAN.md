# VEO Digital Twin - Post-Processing Integration Plan

## A. Current Architecture
The repository currently contains parallel, un-integrated workflows merged from `main`, `knowledge-base`, and `synthetic-data-and-scripts` branches:
1. **E57 & Vision Localization**: `extract_e57.py` extracts images/poses to a centralized SQLite database (`dataset/veo_localization.db`). `build_embeddings.py` and `localize.py` use DINOv2 for viewpoint matching.
2. **Label Detection (Monolith)**: `synthetic_pipeline/find_labels.py` reads the E57 directly (bypassing the SQLite DB), detects orange tape, runs RapidOCR, projects pixels to 3D point clouds, and outputs `tags.csv`.
3. **Knowledge Base**: `scripts/ingest_docs.py` parses PDFs (e.g., ABB manuals) into Markdown chunks and a separate FTS-enabled SQLite database (`data/knowledge.db`).

## B. Existing Components & Responsibilities
- **Reusable Components**:
  - `extract_e57.py` & `db/database.py`: The robust foundational extraction layer. 
  - `build_embeddings.py` & `localize.py`: Camera/viewpoint localization subsystem (retained as requested).
  - `e57/geometry.py`: Quaternion and camera forward vector math.
  - OCR and orange tape HSV masking logic from `find_labels.py`.
  - 3D raycasting and projection math (`project` and `intrinsics` functions) from `find_labels.py`.
  - `ingest_docs.py` logic for parsing manuals.
- **Obsolete/Duplicate Components**:
  - `synthetic_pipeline/extract_images.py`: Obsolete (replaced by `extract_e57.py`).
  - `synthetic_pipeline/inspect_e57.py`: Duplicate of `inspect_dataset.py`.
  - `synthetic_pipeline/find_labels.py`: Must be dismantled. Its logic will be refactored into modular interfaces (`vision/detector.py`, `vision/ocr.py`, `e57/raycast.py`).

## C. ABB Recognition Interface (To Be Exposed)
The object recognition logic must be decoupled from the pipeline. We will define an abstract interface in `vision/detector.py`:
```python
def detect_objects(image_array) -> list[dict]:
    # Returns list of dictionaries:
    # {
    #     "label": str,
    #     "class": str,
    #     "confidence": float,
    #     "bounding_box": [x_min, y_min, x_max, y_max]
    # }
```
*Note: A mock implementation will be used in demo mode until the real ABB detector is ready.*

## D. Data Schema for Final Asset/Tag Database
The existing `veo_localization.db` will be expanded to the final unified database.
- **scans**: `id`, `guid`, `name`, `x, y, z`, `qw, qx, qy, qz`
- **images**: `id`, `scan_id`, `file_path`, dimensions, pos/rot, `projection_type`
- **cubicles**: `id`, `name`, `location_x, location_y, location_z`
- **assets**: `id`, `cubicle_id`, `type`, `manufacturer`, `model`
- **detections**: `id`, `image_id`, `bbox`, `class`, `confidence`
- **documents**: `id`, `asset_id` (or tag_id), `file_path`, `title`
- **tags**: `tag_id`, `parent_tag_id`, `label`, `asset_type`, `description`, `x, y, z`, `anchor_x, y, z`, `source_scan_id`, `source_image_id`, `confidence`

## E. E57 → Database Pipeline
The unified pipeline (`process_e57.py`) will orchestrate the flow:
1. `extract_e57.py` populates `scans` and `images` tables.
2. `build_embeddings.py` indexes viewpoints.
3. The detector iterates over extracted `images` in the DB and populates the `detections` table.
4. `vision/ocr.py` refines detections with text labels.
5. `e57/raycast.py` converts 2D `detections` to 3D anchors, creating `tags` and `assets`.
6. Document matching links PDFs to `assets`.
7. `exporters/` generate JSON/CSV packages.

## F. How ABB Recognition Consumes Images
The detector will **not** parse the E57 file. It will query the `images` table from the SQLite database, load the extracted JPEG from disk, run inference, and insert results into the `detections` table.

## G. Attaching Localization to ABB Detections
Instead of repeating point-cloud lookups, the new `e57/raycast.py` module will:
1. Receive a 2D bounding box from the `detections` table.
2. Load the corresponding camera intrinsics and 3D pose (`x, y, z, qw, qx, qy, qz`) from the `images` table.
3. Fetch only the local point cloud for the parent `scan_id` from the E57 file.
4. Raycast the 2D pixel center to calculate the 3D XYZ anchor and attach it to the `tag` and `asset`.

## H. Recommended Interface for FastAPI Backend
The FastAPI frontend should **not** interact directly with the E57 file or the Python scripts. 
- It should mount the exported `outputs/` folder.
- It will read `tags.json` and `assets.json` to populate the frontend UI.
- It will query the SQLite database via read-only endpoints (e.g., `GET /api/tags`, `GET /api/images/{id}`) to display images and localized bounding boxes dynamically.

## I. What Should NOT Be Implemented (Handled by Teammates)
1. **The Actual ABB Object Detector model**: We will only build the Python interface (`vision/detector.py`) and a mock fallback.
2. **The FastAPI Frontend/Backend**: We will only provide the SQLite DB and export adapters (JSON/CSV) that the API will consume.
3. **FAISS/PCA/Large-scale Optimization**: Strict E57 parsing and basic SQLite/DINOv2 arrays are sufficient for the hackathon scope.

## J. Minimal Next Implementation Steps for Demo
1. **Database Update**: Execute schema migrations in `db/database.py` to add tables: `cubicles`, `assets`, `detections`, `tags`.
2. **Refactoring Vision**: Extract OCR from `find_labels.py` into `vision/ocr.py`. Create `vision/detector.py` with a mock detector returning bounding boxes.
3. **Refactoring Geometry**: Move the `project()`, `intrinsics()`, and E57 point-cloud loading logic from `find_labels.py` into `e57/raycast.py`.
4. **Document Integration**: Write a simple adapter that matches an OCR label (e.g., `REX615`) to the docs in `knowledge.db`.
5. **Orchestration**: Write `process_e57.py` and `exporters/generic_json.py` to tie it all together into a final `outputs/tags.json` export package.
