import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional
from src.utils import polygon_to_bbox, get_bbox_center, get_bbox_height, get_bbox_width

logger = logging.getLogger("paddleocr-table-to-csv")


class OCREngine:
    """Wrapper around PaddleOCR engine for text detection and recognition."""

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def _init_ocr(self):
        """Lazy load PaddleOCR instance."""
        if self._ocr is None:
            logger.info(f"Initializing PaddleOCR (lang={self.lang}, gpu={self.use_gpu})...")
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 2.7+ params
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    enable_mkldnn=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise RuntimeError(
                    "PaddleOCR initialization failed. Ensure paddlepaddle and paddleocr packages are installed properly."
                ) from e

    def detect_text(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5,
        ocr_instance: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Run OCR on an image and return structured results.
        If ocr_instance is provided (e.g. mock during testing), use that instead.
        """
        if ocr_instance is not None:
            engine = ocr_instance
        else:
            self._init_ocr()
            engine = self._ocr

        logger.info("Running PaddleOCR text detection...")
        try:
            if hasattr(engine, 'ocr'):
                raw_results = engine.ocr(image)
            elif hasattr(engine, 'predict'):
                raw_results = engine.predict(image)
            else:
                raw_results = engine(image)
        except Exception as e:
            logger.error(f"OCR execution error: {e}")
            return []

        processed_results: List[Dict[str, Any]] = []

        if not raw_results or raw_results[0] is None:
            logger.warning("No text detected by OCR engine.")
            return []

        # Handle PaddleOCR output formats (dict with res / dt_polys or list)
        lines = []
        if isinstance(raw_results, list) and len(raw_results) > 0:
            item = raw_results[0]
            if isinstance(item, dict):
                # Standard PaddleX / PaddleOCR dict structure
                res = item.get('res', item)
                polys = res.get('dt_polys', res.get('boxes', []))
                texts = res.get('rec_text', res.get('texts', []))
                scores = res.get('rec_score', res.get('scores', []))
                for p, t, s in zip(polys, texts, scores):
                    lines.append([p, (t, s)])
            elif isinstance(item, list):
                lines = item
            else:
                lines = raw_results
        elif isinstance(raw_results, dict):
            res = raw_results.get('res', raw_results)
            polys = res.get('dt_polys', res.get('boxes', []))
            texts = res.get('rec_text', res.get('texts', []))
            scores = res.get('rec_score', res.get('scores', []))
            for p, t, s in zip(polys, texts, scores):
                lines.append([p, (t, s)])

        for line in lines:
            if not line or len(line) < 2:
                continue
            poly, (text, conf) = line[0], line[1]
            text_str = str(text).strip()
            conf_val = float(conf)

            if conf_val < min_confidence or not text_str:
                continue

            bbox = polygon_to_bbox(poly)
            center = get_bbox_center(bbox)
            height = get_bbox_height(bbox)
            width = get_bbox_width(bbox)

            processed_results.append({
                "text": text_str,
                "confidence": conf_val,
                "polygon": poly,
                "bbox": bbox,       # (xmin, ymin, xmax, ymax)
                "center": center,   # (cx, cy)
                "height": height,
                "width": width
            })

        logger.info(f"OCR detected {len(processed_results)} text blocks (confidence >= {min_confidence}).")
        return processed_results
