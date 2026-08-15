import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("paddleocr-table-to-csv")


class ImagePreprocessor:
    """Handles image preprocessing for OCR optimization."""

    def __init__(self, target_min_dim: int = 1000):
        self.target_min_dim = target_min_dim

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from disk using OpenCV, handling unicode paths if necessary."""
        img = cv2.imread(image_path)
        if img is None:
            # Fallback for unicode file paths in OpenCV on Windows
            try:
                img_array = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to read image at {image_path}: {e}")
                img = None
        if img is None:
            raise ValueError(f"Could not read or decode image file: {image_path}")
        return img

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect skew angle and rotate image to align horizontally."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Binarize for line detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find non-zero points (text pixels)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image

        # Get minimum area rectangle covering non-zero points
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Adjust angle to [-45, 45] range
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Ignore tiny angles
        if abs(angle) < 0.5 or abs(angle) > 30:
            return image

        logger.info(f"Deskewing image by angle: {angle:.2f} degrees")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def upscale_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Upscale image if its minimum dimension is less than target_min_dim."""
        h, w = image.shape[:2]
        min_dim = min(h, w)
        if min_dim < self.target_min_dim and min_dim > 0:
            scale = float(self.target_min_dim) / float(min_dim)
            scale = min(scale, 3.0)  # Max upscale 3x
            new_w = int(w * scale)
            new_h = int(h * scale)
            upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Upscaled image from ({w}x{h}) to ({new_w}x{new_h}), scale={scale:.2f}")
            return upscaled, scale
        return image, 1.0

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to improve contrast."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply light denoising."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 5, 7, 21)

    def preprocess(self, image_path: str, deskew: bool = True, enhance: bool = True) -> Tuple[np.ndarray, float]:
        """
        Main preprocessing entrypoint.
        Returns:
            processed_image: BGR numpy image ready for OCR / Line Detection
            scale_factor: scale factor applied during upscaling
        """
        img = self.load_image(image_path)

        if deskew:
            try:
                img = self.deskew(img)
            except Exception as e:
                logger.warning(f"Deskew failed: {e}")

        if enhance:
            try:
                img = self.enhance_contrast(img)
            except Exception as e:
                logger.warning(f"Contrast enhancement failed: {e}")

        img, scale_factor = self.upscale_if_needed(img)
        return img, scale_factor
