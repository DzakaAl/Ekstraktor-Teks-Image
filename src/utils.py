import os
import numpy as np
from typing import List, Dict, Any, Tuple


def polygon_to_bbox(poly: List[List[float]]) -> Tuple[int, int, int, int]:
    """
    Convert a 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    to axis-aligned bbox (xmin, ymin, xmax, ymax).
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate center (x, y) of bbox (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def get_bbox_height(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate height of bbox (xmin, ymin, xmax, ymax)."""
    return bbox[3] - bbox[1]


def get_bbox_width(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate width of bbox (xmin, ymin, xmax, ymax)."""
    return bbox[2] - bbox[0]


def calculate_vertical_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate vertical overlap ratio relative to the smaller box height."""
    _, y1_min, _, y1_max = box1
    _, y2_min, _, y2_max = box2

    overlap_min = max(y1_min, y2_min)
    overlap_max = min(y1_max, y2_max)

    if overlap_max <= overlap_min:
        return 0.0

    overlap_height = overlap_max - overlap_min
    min_h = min(y1_max - y1_min, y2_max - y2_min)
    if min_h <= 0:
        return 0.0

    return overlap_height / float(min_h)


def calculate_horizontal_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate horizontal overlap ratio relative to the smaller box width."""
    x1_min, _, x1_max, _ = box1
    x2_min, _, x2_max, _ = box2

    overlap_min = max(x1_min, x2_min)
    overlap_max = min(x1_max, x2_max)

    if overlap_max <= overlap_min:
        return 0.0

    overlap_width = overlap_max - overlap_min
    min_w = min(x1_max - x1_min, x2_max - x2_min)
    if min_w <= 0:
        return 0.0

    return overlap_width / float(min_w)


def draw_debug_boxes(image: np.ndarray, ocr_results: List[Dict[str, Any]], output_path: str):
    """Draw OCR bounding boxes with text annotations and confidence."""
    import cv2
    img_copy = image.copy()
    for item in ocr_results:
        bbox = item['bbox']
        text = item['text']
        conf = item['confidence']
        xmin, ymin, xmax, ymax = bbox

        cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{text} ({conf:.2f})"
        cv2.putText(img_copy, label, (xmin, max(15, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_copy)


def draw_rows_cols(image: np.ndarray, rows: List[List[Dict[str, Any]]], col_boundaries: List[Tuple[int, int]], output_path: str):
    """Draw detected rows and column boundaries in distinct colors."""
    import cv2
    img_copy = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (0, 128, 128), (128, 128, 0)
    ]

    # Draw rows
    for idx, row in enumerate(rows):
        color = colors[idx % len(colors)]
        for item in row:
            xmin, ymin, xmax, ymax = item['bbox']
            cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_copy, f"R{idx}", (xmin, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw column vertical boundaries
    h = img_copy.shape[0]
    for c_idx, (c_start, c_end) in enumerate(col_boundaries):
        cv2.line(img_copy, (c_start, 0), (c_start, h), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.line(img_copy, (c_end, 0), (c_end, h), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(img_copy, f"C{c_idx}", (c_start + 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_copy)

