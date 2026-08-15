import numpy as np
import logging
from typing import List, Dict, Any
from src.utils import calculate_vertical_overlap

logger = logging.getLogger("paddleocr-table-to-csv")


class RowDetector:
    """Groups OCR text blocks into rows based on adaptive Y-coordinate clustering."""

    def __init__(self, tolerance_factor: float = 0.5, min_overlap_ratio: float = 0.4):
        self.tolerance_factor = tolerance_factor
        self.min_overlap_ratio = min_overlap_ratio

    def detect_rows(self, ocr_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group OCR text blocks into sorted rows using adaptive Y clustering.
        Returns:
            rows: List of rows, where each row is a list of OCR items sorted left-to-right (by xmin).
        """
        if not ocr_results:
            return []

        # Sort OCR results primarily by Y-top / Y-center
        sorted_items = sorted(ocr_results, key=lambda item: item['bbox'][1])

        # Compute median text height across all bounding boxes
        heights = [item['height'] for item in sorted_items if item['height'] > 0]
        median_height = float(np.median(heights)) if heights else 15.0
        row_tolerance = max(8.0, median_height * self.tolerance_factor)

        logger.info(f"Median text height: {median_height:.2f}px, adaptive row tolerance: {row_tolerance:.2f}px")

        rows: List[List[Dict[str, Any]]] = []

        for item in sorted_items:
            bbox = item['bbox']
            cy = item['center'][1]

            assigned = False
            for row in rows:
                # Calculate average Y center of existing row
                row_cys = [r_item['center'][1] for r_item in row]
                avg_row_cy = float(np.mean(row_cys))

                # Check vertical overlap with items in the row
                has_high_overlap = any(
                    calculate_vertical_overlap(bbox, r_item['bbox']) >= self.min_overlap_ratio
                    for r_item in row
                )

                if abs(cy - avg_row_cy) <= row_tolerance or has_high_overlap:
                    row.append(item)
                    assigned = True
                    break

            if not assigned:
                rows.append([item])

        # Sort rows top-to-bottom by average Y-center
        rows.sort(key=lambda r: np.mean([item['center'][1] for item in r]))

        # Sort items inside each row left-to-right by xmin
        for r_idx in range(len(rows)):
            rows[r_idx].sort(key=lambda item: item['bbox'][0])

        logger.info(f"Detected {len(rows)} rows.")
        return rows
