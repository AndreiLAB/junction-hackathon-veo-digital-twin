from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Detection:
    image_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    ocr_text: Optional[str] = None
    ocr_conf: Optional[float] = None

@dataclass
class DocumentMatch:
    document_id: int
    title: str
    section_path: str
    page_start: int
    text_snippet: str

@dataclass
class LocalizedDetection:
    detection: Detection
    scan_id: int
    camera_position: Tuple[float, float, float]
    camera_orientation: Tuple[float, float, float, float]
    estimated_world_position: Optional[Tuple[float, float, float]] = None

@dataclass
class Tag:
    tag_id: str
    parent_tag_id: Optional[str]
    label: str
    asset_type: str
    confidence: float
    x: float
    y: float
    z: float
    bbox: Tuple[float, float, float, float]
    source_image_id: int
    source_scan_id: int
    physical_tag_id: Optional[str] = None
    documents: List[DocumentMatch] = field(default_factory=list)

@dataclass
class Asset:
    asset_id: str
    cubicle_id: Optional[str]
    type: str
    manufacturer: str
    model: str

@dataclass
class PhysicalTag:
    tag_id: str
    parent_tag_id: Optional[str]
    label: str
    asset_type: str
    confidence: float
    x: float
    y: float
    z: float
    observation_count: int
    source_images: List[int]
    source_scans: List[int]
    source_observations: List[dict] = field(default_factory=list)
    documents: List[DocumentMatch] = field(default_factory=list)
