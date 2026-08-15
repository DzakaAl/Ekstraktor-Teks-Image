import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("paddleocr-table-to-csv")


class TableDetector:
    """Detects table structure (grid lines vs gridless) using OpenCV computer vision."""

    def __init__(self, min_line_length_ratio: float = 0.02):
        self.min_line_length_ratio = min_line_length_ratio

    def detect_table_type_and_lines(
        self, image: np.ndarray, debug_dir: Optional[str] = None
    ) -> Tuple[str, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """
        Detect whether the image contains grid lines and extract horizontal/vertical line masks.
        Returns:
            table_type: "lined" or "gridless"
            horizontal_lines: contours/masks of horizontal lines
            vertical_lines: contours/masks of vertical lines
            grid_mask: combined line mask image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Adaptive threshold / binarization
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        h, w = gray.shape[:2]

        # Morphological kernel length based on image dimensions
        horiz_kernel_len = max(15, int(w * self.min_line_length_ratio))
        vert_kernel_len = max(15, int(h * self.min_line_length_ratio))

        # Detect horizontal lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
        horiz_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

        # Detect vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
        vert_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

        # Combine horizontal and vertical lines
        table_grid = cv2.addWeighted(horiz_mask, 1.0, vert_mask, 1.0, 0.0)

        # Count line pixels to determine if image is lined or gridless
        horiz_pixels = np.sum(horiz_mask > 0)
        vert_pixels = np.sum(vert_mask > 0)
        total_pixels = h * w

        is_lined = (horiz_pixels / total_pixels > 0.005) and (vert_pixels / total_pixels > 0.005)
        table_type = "lined" if is_lined else "gridless"

        logger.info(f"Table type detected: {table_type.upper()} (horiz_ratio: {horiz_pixels/total_pixels:.4f}, vert_ratio: {vert_pixels/total_pixels:.4f})")

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "detected_horizontal_lines.png"), horiz_mask)
            cv2.imwrite(os.path.join(debug_dir, "detected_vertical_lines.png"), vert_mask)
            cv2.imwrite(os.path.join(debug_dir, "detected_table_grid.png"), table_grid)

        # Find line contours
        h_contours, _ = cv2.findContours(horiz_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vert_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return table_type, h_contours, v_contours, table_grid

    def extract_grid_cells(
        self, image: np.ndarray, horiz_mask: np.ndarray, vert_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Extract cell bounding boxes (xmin, ymin, xmax, ymax) from intersecting lines.
        """
        table_grid = cv2.addWeighted(horiz_mask, 1.0, vert_mask, 1.0, 0.0)
        intersections = cv2.bitwise_and(horiz_mask, vert_mask)

        # Find contours of grid cells
        contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out tiny or huge contours
            if w > 20 and h > 10 and w < image.shape[1] * 0.95 and h < image.shape[0] * 0.95:
                cell_boxes.append((x, y, x + w, y + h))

        return cell_boxes
