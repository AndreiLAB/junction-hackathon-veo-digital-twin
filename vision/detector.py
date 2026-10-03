from models import Detection
from typing import List

class DetectorInterface:
    def detect(self, image_array, image_id: int) -> List[Detection]:
        raise NotImplementedError

class MockDetector(DetectorInterface):
    """
    Mock detector for ABB relays.
    Returns a dummy bounding box in the center of the image.
    """
    def detect(self, image_array, image_id: int) -> List[Detection]:
        h, w = image_array.shape[:2]
        cx, cy = w // 2, h // 2
        # Mock bounding box with fixed size 100x100
        x0, y0 = cx - 50, cy - 50
        x1, y1 = cx + 50, cy + 50
        
        return [
            Detection(
                image_id=image_id,
                class_name="abb_relion_615",
                confidence=0.92,
                bbox=(x0, y0, x1, y1)
            )
        ]

class ABBDetector(DetectorInterface):
    """
    Real ABB detector stub (to be implemented by ML teammate).
    """
    def detect(self, image_array, image_id: int) -> List[Detection]:
        # TODO: Implement YOLO or other object detector here
        # return class_name, confidence, bbox
        return []

def get_detector(backend_name: str) -> DetectorInterface:
    if backend_name == "abb":
        return ABBDetector()
    elif backend_name == "mock":
        return MockDetector()
    else:
        raise ValueError(f"Unknown detector backend: {backend_name}")
