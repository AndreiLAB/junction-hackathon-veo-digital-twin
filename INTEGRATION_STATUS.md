# VEO Integration Status

## Existing Components Reused
- `extract_e57.py`: Unmodified E57 parsing and SQLite DB population.
- `db/database.py`: Existing connection logic and array adapters, extended with new tables.
- `scripts/kb_search.py`: Core knowledge base FTS5 searching algorithm.
- `backend/`: FastAPI backend and Pydantic schemas (DetectionIn, CabinetIn) dictate our output format.
- `build_embeddings.py` & `localize.py`: (Intact, can be plugged back if view-localization is needed).

## Files Refactored
- `synthetic_pipeline/find_labels.py`: Dismantled entirely. Its responsibilities were distributed across:
  - `vision/label_detector.py` (Orange tape logic)
  - `vision/ocr.py` (RapidOCR wrapper)
  - `e57/raycast.py` (3D projection and point-cloud intersection math)
  - `exporters/` (JSON/CSV dumping)

## New Files Created
- `models.py`: Central data structures (Detection, Tag, Asset, etc.).
- `vision/detector.py`: Detector interface containing `MockDetector` (returns hardcoded ABB relay bounding box).
- `vision/label_detector.py`: Specific detection for orange tape cabinet labels.
- `vision/ocr.py`: Isolated RapidOCR module.
- `e57/raycast.py`: Reusable 2D to 3D point cloud intersection.
- `tagging/tagger.py`: Logic combining a localized detection with KB docs into a `Tag`.
- `knowledge/adapter.py`: Adapter wrapping `scripts.kb_search` to map equipment classes to manuals.
- `exporters/json_exporter.py` & `csv_exporter.py`: Outputs formatted for the backend.
- `process_e57.py`: The central orchestrator uniting all steps.

## Pipeline Diagram
```
cloud.e57
   ↓
extract_e57.py  → (dataset/veo_localization.db)
   ↓
vision detectors (Mock ABB + Label/OCR)
   ↓
e57/raycast.py (Intersect 2D Box with E57 Point Cloud)
   ↓
knowledge/adapter.py (Match to manuals)
   ↓
tagging/tagger.py (Assemble final Tag records)
   ↓
exporters/
   ↓
outputs/tags.json & outputs/tags.csv
```

## Data Flow & API Contract
- Output JSON `tags.json` is mapped to the FastAPI backend expectations:
  ```json
  [
    {
      "tag_id": "uuid",
      "parent_tag_id": null,
      "label": "H04-SOLAR-1",
      "type": "cubicle",
      "position": {"x": 1.0, "y": 2.0, "z": 3.0},
      "confidence": 0.95,
      "documents": []
    },
    {
      "tag_id": "uuid",
      "parent_tag_id": "parent-uuid",
      "label": "abb_relion_615",
      "type": "equipment",
      "position": {"x": 1.1, "y": 2.1, "z": 3.1},
      "confidence": 0.92,
      "documents": [
         {
           "title": "ABB 615 Manual",
           "section": "Overview",
           "page": 10,
           "snippet": "..."
         }
      ]
    }
  ]
  ```

## Current Limitations
- **Mock Detector**: `vision/detector.py` currently returns a dummy bounding box for `abb_relion_615`. Once the teammate finishes the actual ML model, it simply drops into this interface.
- **Focal Length Approximation**: When `pinholeRepresentation.focalLength` is missing in E57, `raycast.py` currently approximates it as `W/2`.

## Execution Command
To run the end-to-end processor:
```bash
python process_e57.py --e57 /path/to/cloud.e57
```
