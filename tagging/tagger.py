import uuid
from models import Tag, LocalizedDetection, DocumentMatch
from typing import List, Optional

def generate_tag(loc_det: LocalizedDetection, parent_id: Optional[str] = None, docs: List[DocumentMatch] = None) -> Tag:
    """
    Generates a unified Tag from a localized detection.
    """
    if docs is None:
        docs = []
        
    det = loc_det.detection
    tag_id = str(uuid.uuid4())
    
    # If the detection is a cabinet label, the asset type is "cubicle"
    # Otherwise it's an "equipment"
    if det.class_name == "cabinet_label":
        asset_type = "cubicle"
        label = det.ocr_text if det.ocr_text else "UNKNOWN_CABINET"
    else:
        asset_type = "equipment"
        label = det.class_name
        
    x, y, z = loc_det.estimated_world_position if loc_det.estimated_world_position else (0.0, 0.0, 0.0)
    
    return Tag(
        tag_id=tag_id,
        parent_tag_id=parent_id,
        label=label,
        asset_type=asset_type,
        confidence=det.confidence,
        x=x,
        y=y,
        z=z,
        bbox=det.bbox,
        source_image_id=det.image_id,
        source_scan_id=loc_det.scan_id,
        documents=docs
    )
