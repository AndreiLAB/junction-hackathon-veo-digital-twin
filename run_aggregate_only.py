import json
import csv
from models import Tag, DocumentMatch
from tagging.aggregator import aggregate_tags
from exporters.json_exporter import export_physical_tags_to_json
from exporters.csv_exporter import export_physical_tags_to_csv

def main():
    # Read CSV for complete metadata including image/scan IDs
    tags = []
    with open("outputs/tags.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append(Tag(
                tag_id=row["tag_id"],
                parent_tag_id=row["parent_tag_id"] if row["parent_tag_id"] else None,
                label=row["label"],
                asset_type=row["type"],
                confidence=float(row["confidence"]),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                source_image_id=int(row["source_image"]),
                source_scan_id=int(row["source_scan"]),
                documents=[]
            ))
            
    physical_tags = aggregate_tags(tags, dist_threshold=1.5)
    print(f"Raw detections: {len(tags)} -> Unique physical tags: {len(physical_tags)}")
    
    export_physical_tags_to_json(physical_tags, "outputs/physical_tags.json")
    export_physical_tags_to_csv(physical_tags, "outputs/physical_tags.csv")

if __name__ == "__main__":
    main()
