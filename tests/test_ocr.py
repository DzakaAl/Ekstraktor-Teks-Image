import unittest
import numpy as np
from src.ocr_engine import OCREngine


class MockPaddleOCR:
    def ocr(self, img, cls=True):
        # Simulated raw PaddleOCR output
        return [[
            [[[10, 10], [50, 10], [50, 30], [10, 30]], ("Header1", 0.95)],
            [[[60, 10], [100, 10], [100, 30], [60, 30]], ("Header2", 0.90)],
            [[[10, 40], [50, 40], [50, 60], [10, 60]], ("081234", 0.85)],
            [[[60, 40], [100, 40], [100, 60], [60, 60]], ("LowConf", 0.20)],
        ]]


class TestOCREngine(unittest.TestCase):
    def test_ocr_engine_detection(self):
        engine = OCREngine(lang="en", use_gpu=False)
        mock_ocr = MockPaddleOCR()
        dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)

        # Filter with min_confidence = 0.5
        results = engine.detect_text(dummy_img, min_confidence=0.5, ocr_instance=mock_ocr)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['text'], "Header1")
        self.assertEqual(results[1]['text'], "Header2")
        self.assertEqual(results[2]['text'], "081234")

        # Verify confidence thresholding
        low_conf_items = [r for r in results if r['text'] == "LowConf"]
        self.assertEqual(len(low_conf_items), 0)


if __name__ == "__main__":
    unittest.main()

