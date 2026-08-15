import unittest
import numpy as np
import pandas as pd
from src.row_detector import RowDetector
from src.column_detector import ColumnDetector
from src.table_parser import TableParser


class TestTableParser(unittest.TestCase):
    def test_row_detector_grouping(self):
        row_detector = RowDetector(tolerance_factor=0.5)

        ocr_items = [
            {"text": "No", "bbox": (10, 10, 30, 30), "center": (20, 20), "height": 20, "width": 20},
            {"text": "Nama", "bbox": (50, 12, 90, 28), "center": (70, 20), "height": 16, "width": 40},
            {"text": "1", "bbox": (10, 50, 25, 70), "center": (17.5, 60), "height": 20, "width": 15},
            {"text": "Budi", "bbox": (50, 52, 85, 68), "center": (67.5, 60), "height": 16, "width": 35},
        ]

        rows = row_detector.detect_rows(ocr_items)
        self.assertEqual(len(rows), 2)
        self.assertEqual([item['text'] for item in rows[0]], ["No", "Nama"])
        self.assertEqual([item['text'] for item in rows[1]], ["1", "Budi"])

    def test_column_detector(self):
        column_detector = ColumnDetector()

        rows = [
            [
                {"text": "No", "bbox": (10, 10, 30, 30), "center": (20, 20), "height": 20, "width": 20},
                {"text": "Nama", "bbox": (100, 10, 150, 30), "center": (125, 20), "height": 20, "width": 50},
            ],
            [
                {"text": "1", "bbox": (10, 50, 20, 70), "center": (15, 60), "height": 20, "width": 10},
                {"text": "Budi", "bbox": (100, 50, 140, 70), "center": (120, 60), "height": 20, "width": 40},
            ]
        ]

        cols = column_detector.detect_columns(rows, image_width=200)
        self.assertEqual(len(cols), 2)

    def test_header_sanitization(self):
        parser = TableParser()
        headers = ["Nama", "Nama", "", "Telepon"]
        sanitized = parser.sanitize_headers(headers)
        self.assertEqual(sanitized, ["Nama", "Nama_1", "column_3", "Telepon"])


if __name__ == "__main__":
    unittest.main()
