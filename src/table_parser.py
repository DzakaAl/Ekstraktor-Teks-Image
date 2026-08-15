import os
import cv2
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional

from src.image_preprocessor import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.table_detector import TableDetector
from src.row_detector import RowDetector
from src.column_detector import ColumnDetector
from src.utils import draw_debug_boxes, draw_rows_cols

logger = logging.getLogger("paddleocr-table-to-csv")


class TableParser:
    """End-to-end table reconstruction parser."""

    def __init__(self, lang: str = "en", min_confidence: float = 0.5, use_gpu: bool = False):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(lang=lang, use_gpu=use_gpu)
        self.table_detector = TableDetector()
        self.row_detector = RowDetector()
        self.column_detector = ColumnDetector()
        self.min_confidence = min_confidence

    def is_header_row(self, first_row_texts: List[str]) -> bool:
        """Heuristic check to determine if the first row is a header row."""
        if not first_row_texts:
            return False

        non_empty = [t for t in first_row_texts if t.strip()]
        if not non_empty:
            return False

        # If most items in first row are numeric, it's probably data, not a header
        numeric_count = sum(1 for t in non_empty if t.replace('.', '').replace(',', '').isdigit())
        if numeric_count / len(non_empty) > 0.6:
            return False

        return True

    def sanitize_headers(self, headers: List[str]) -> List[str]:
        """Ensure column headers are clean and unique."""
        sanitized = []
        counts: Dict[str, int] = {}

        for idx, h in enumerate(headers):
            clean_name = h.strip()
            if not clean_name:
                clean_name = f"column_{idx + 1}"

            if clean_name in counts:
                counts[clean_name] += 1
                unique_name = f"{clean_name}_{counts[clean_name]}"
            else:
                counts[clean_name] = 0
                unique_name = clean_name

            sanitized.append(unique_name)

        return sanitized

    def parse_image(
        self,
        image_path: str,
        debug_dir: Optional[str] = None,
        ocr_instance: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract table from image file and convert to Pandas DataFrame.
        Returns:
            df: Pandas DataFrame containing reconstructed table
            stats: Dictionary of execution statistics
        """
        logger.info(f"Processing image: {image_path}")

        # 1. Preprocess image
        processed_img, scale_factor = self.preprocessor.preprocess(image_path)
        img_h, img_w = processed_img.shape[:2]

        if debug_dir:
            orig_dir = os.path.join(debug_dir, "original")
            prep_dir = os.path.join(debug_dir, "preprocessed")
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(prep_dir, exist_ok=True)
            
            orig_img = self.preprocessor.load_image(image_path)
            base_name = os.path.basename(image_path)
            cv2.imwrite(os.path.join(orig_dir, base_name), orig_img)
            cv2.imwrite(os.path.join(prep_dir, base_name), processed_img)

        # 2. Run OCR
        ocr_results = self.ocr_engine.detect_text(
            processed_img,
            min_confidence=self.min_confidence,
            ocr_instance=ocr_instance
        )

        if debug_dir:
            ocr_debug_dir = os.path.join(debug_dir, "ocr")
            draw_debug_boxes(processed_img, ocr_results, os.path.join(ocr_debug_dir, os.path.basename(image_path)))

        if not ocr_results:
            logger.warning(f"No valid OCR text found in {image_path}")
            return pd.DataFrame(), {
                "text_blocks": 0, "rows": 0, "cols": 0, "cells": 0, "avg_confidence": 0.0
            }

        # Calculate average confidence
        avg_conf = float(np.mean([item['confidence'] for item in ocr_results])) * 100.0

        # 3. Detect table type
        lines_debug_dir = os.path.join(debug_dir, "detected_lines") if debug_dir else None
        table_type, h_contours, v_contours, _ = self.table_detector.detect_table_type_and_lines(
            processed_img, debug_dir=lines_debug_dir
        )

        # 4. Group rows
        rows = self.row_detector.detect_rows(ocr_results)

        # 5. Detect global columns
        col_boundaries = self.column_detector.detect_columns(rows, img_w)

        if debug_dir:
            cells_debug_dir = os.path.join(debug_dir, "detected_cells")
            draw_rows_cols(
                processed_img, rows, col_boundaries,
                os.path.join(cells_debug_dir, os.path.basename(image_path))
            )

        # 6. Assign items to cells in grid matrix
        num_rows = len(rows)
        num_cols = len(col_boundaries)
        grid: List[List[List[str]]] = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

        for r_idx, row in enumerate(rows):
            for item in row:
                cx = item['center'][0]
                text = item['text']

                # Find matching column by X center
                col_idx = 0
                assigned_col = False
                for c_i, (c_start, c_end) in enumerate(col_boundaries):
                    if c_start <= cx <= c_end:
                        col_idx = c_i
                        assigned_col = True
                        break
                if not assigned_col:
                    # Find closest column boundary
                    col_idx = min(
                        range(num_cols),
                        key=lambda c_i: min(abs(cx - col_boundaries[c_i][0]), abs(cx - col_boundaries[c_i][1]))
                    )

                grid[r_idx][col_idx].append(text)

        # Join cell fragments with spaces
        table_data: List[List[str]] = []
        for r in range(num_rows):
            row_data = []
            for c in range(num_cols):
                cell_text = " ".join(grid[r][c]).strip()
                row_data.append(cell_text)
            table_data.append(row_data)

        # 7. Extract header and data rows
        if table_data and self.is_header_row(table_data[0]):
            header_row = table_data[0]
            data_rows = table_data[1:]
        else:
            header_row = [f"column_{c + 1}" for c in range(num_cols)]
            data_rows = table_data

        clean_headers = self.sanitize_headers(header_row)

        df = pd.DataFrame(data_rows, columns=clean_headers, dtype=str)

        stats = {
            "text_blocks": len(ocr_results),
            "rows": num_rows,
            "cols": num_cols,
            "cells": num_rows * num_cols,
            "avg_confidence": avg_conf
        }

        logger.info(
            f"Table parse complete: {num_rows} rows, {num_cols} cols, "
            f"confidence: {avg_conf:.1f}%"
        )
        return df, stats
