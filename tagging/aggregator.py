import numpy as np
import uuid
from typing import List
from models import Tag, PhysicalTag

def aggregate_tags(tags: List[Tag], dist_threshold=1.5) -> List[PhysicalTag]:
    """
    Groups repeated detections of the same label if they are spatially close.
    """
    groups = {}
    for t in tags:
        key = (t.label, t.asset_type)
        groups.setdefault(key, []).append(t)
        
    physical_tags = []
    
    for (label, asset_type), group_tags in groups.items():
        points = np.array([[t.x, t.y, t.z] for t in group_tags])
        n = len(points)
        
        # 1. Compute distance adjacency
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < dist_threshold:
                    adj[i, j] = True
                    adj[j, i] = True
                    
        # 2. Find connected components
        visited = set()
        components = []
        for i in range(n):
            if i not in visited:
                comp = []
                stack = [i]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        comp.append(curr)
                        for neighbor in range(n):
                            if adj[curr, neighbor] and neighbor not in visited:
                                stack.append(neighbor)
                components.append(comp)
                
        # 3. Create PhysicalTag for each cluster
        for comp in components:
            comp_tags = [group_tags[i] for i in comp]
            
            pts = np.array([[t.x, t.y, t.z] for t in comp_tags])
            confs = np.array([t.confidence for t in comp_tags])
            
            # Robust XYZ via confidence-weighted average
            avg_x, avg_y, avg_z = np.average(pts, axis=0, weights=confs)
            
            source_images = list(set([t.source_image_id for t in comp_tags]))
            source_scans = list(set([t.source_scan_id for t in comp_tags]))
            
            docs = []
            for t in comp_tags:
                if t.documents and not docs:
                    docs = t.documents
            
            ptag = PhysicalTag(
                tag_id=str(uuid.uuid4()),
                parent_tag_id=None, # Will need post-processing to link hierarchies if requested
                label=label,
                asset_type=asset_type,
                confidence=float(np.max(confs)),
                x=float(avg_x),
                y=float(avg_y),
                z=float(avg_z),
                observation_count=len(comp_tags),
                source_images=source_images,
                source_scans=source_scans,
                source_observations=[
                    {"image_id": t.source_image_id, "scan_id": t.source_scan_id, "bbox": t.bbox} for t in comp_tags
                ],
                documents=docs
            )
            physical_tags.append(ptag)
            
            # Back-link raw tags to their aggregate physical tag
            for t in comp_tags:
                t.physical_tag_id = ptag.tag_id
            
    return physical_tags
