import json
from typing import List
from models import Tag

def export_tags_to_json(tags: List[Tag], output_path: str):
    data = []
    for t in tags:
        record = {
            "tag_id": t.tag_id,
            "parent_tag_id": t.parent_tag_id,
            "label": t.label,
            "type": t.asset_type,
            "position": {
                "x": t.x,
                "y": t.y,
                "z": t.z
            },
            "confidence": t.confidence
        }
        if t.documents:
            record["documents"] = [
                {
                    "title": d.title,
                    "section": d.section_path,
                    "page": d.page_start,
                    "snippet": d.text_snippet
                } for d in t.documents
            ]
        data.append(record)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def export_physical_tags_to_json(tags, output_path: str):
    data = []
    for t in tags:
        record = {
            "tag_id": t.tag_id,
            "parent_tag_id": t.parent_tag_id,
            "label": t.label,
            "type": t.asset_type,
            "position": {
                "x": t.x,
                "y": t.y,
                "z": t.z
            },
            "confidence": t.confidence,
            "observation_count": t.observation_count,
            "source_images": t.source_images,
            "source_scans": t.source_scans,
            "source_observations": t.source_observations
        }
        if t.documents:
            record["documents"] = [
                {
                    "title": d.title,
                    "section": d.section_path,
                    "page": d.page_start,
                    "snippet": d.text_snippet
                } for d in t.documents
            ]
        data.append(record)
        
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
