import csv
from typing import List
from models import Tag

def export_tags_to_csv(tags: List[Tag], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # tag_id,parent_tag_id,label,type,x,y,z,confidence,source_image,source_scan
        writer.writerow(["tag_id", "parent_tag_id", "label", "type", "x", "y", "z", "confidence", "source_image", "source_scan"])
        for t in tags:
            writer.writerow([
                t.tag_id,
                t.parent_tag_id if t.parent_tag_id else "",
                t.label,
                t.asset_type,
                t.x,
                t.y,
                t.z,
                t.confidence,
                t.source_image_id,
                t.source_scan_id
            ])

def export_physical_tags_to_csv(tags, output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag_id", "parent_tag_id", "label", "type", "x", "y", "z", "confidence", "observation_count", "source_images", "source_scans"])
        for t in tags:
            writer.writerow([
                t.tag_id,
                t.parent_tag_id if t.parent_tag_id else "",
                t.label,
                t.asset_type,
                t.x,
                t.y,
                t.z,
                t.confidence,
                t.observation_count,
                " ".join(map(str, t.source_images)),
                " ".join(map(str, t.source_scans))
            ])
