from dataclasses import dataclass
from typing import List, Tuple
from rapidocr_onnxruntime import RapidOCR

@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine

def run_ocr(image_array) -> List[OCRResult]:
    ocr = get_ocr_engine()
    res, _ = ocr(image_array)
    results = []
    for box, text, conf in (res or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        results.append(OCRResult(
            text=text,
            confidence=float(conf),
            bbox=(min(xs), min(ys), max(xs), max(ys))
        ))
    return results
