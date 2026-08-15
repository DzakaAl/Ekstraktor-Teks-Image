import numpy as np
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("paddleocr-table-to-csv")


class ColumnDetector:
    """Detects global column boundaries across all rows using horizontal interval clustering and gap analysis."""

    def __init__(self, gap_threshold_factor: float = 0.8):
        self.gap_threshold_factor = gap_threshold_factor

    def detect_columns(
        self, rows: List[List[Dict[str, Any]]], image_width: int
    ) -> List[Tuple[int, int]]:
        """
        Detect horizontal column boundaries (x_min, x_max) across all rows.
        Returns:
            col_boundaries: List of (col_start_x, col_end_x) representing column intervals.
        """
        if not rows:
            return [(0, image_width)]

        all_items = [item for row in rows for item in row]
        if not all_items:
            return [(0, image_width)]

        # Compute median text width
        widths = [item['width'] for item in all_items if item['width'] > 0]
        median_width = float(np.median(widths)) if widths else 30.0

        # Collect horizontal ranges [xmin, xmax] of all text items
        x_ranges = [(item['bbox'][0], item['bbox'][2]) for item in all_items]
        
        # Determine table left and right bounds
        table_left = min(r[0] for r in x_ranges)
        table_right = max(r[1] for r in x_ranges)

        # Build horizontal projection profile (1D histogram of text occupancy)
        proj_length = max(image_width, table_right + 50)
        profile = np.zeros(proj_length, dtype=int)

        for xmin, xmax in x_ranges:
            profile[xmin:xmax] += 1

        # Smooth profile using moving average
        kernel_size = max(5, int(median_width * 0.3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_profile = np.convolve(profile, np.ones(kernel_size)/kernel_size, mode='same')

        # Find gaps (regions between table_left and table_right where smoothed_profile is near 0)
        zero_regions = []
        in_gap = False
        gap_start = 0

        for x in range(table_left, table_right + 1):
            val = smoothed_profile[x]
            if val < 0.2:  # Threshold for gap
                if not in_gap:
                    in_gap = True
                    gap_start = x
            else:
                if in_gap:
                    in_gap = False
                    gap_end = x
                    # Minimum gap width to split columns (adaptive based on median letter/text width)
                    min_gap_width = max(10, int(median_width * self.gap_threshold_factor))
                    if (gap_end - gap_start) >= min_gap_width:
                        zero_regions.append((gap_start, gap_end))

        # Build column intervals from gaps
        col_boundaries: List[Tuple[int, int]] = []
        curr_left = table_left

        for gap_s, gap_e in zero_regions:
            gap_mid = (gap_s + gap_e) // 2
            if gap_mid > curr_left + 10:
                col_boundaries.append((curr_left, gap_mid))
                curr_left = gap_mid

        if curr_left < table_right:
            col_boundaries.append((curr_left, table_right + 20))

        if not col_boundaries:
            col_boundaries = [(table_left, table_right)]

        logger.info(f"Detected {len(col_boundaries)} global columns.")
        return col_boundaries
