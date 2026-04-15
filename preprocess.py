"""
VisionSecure — STAGE 2: Preprocessing & Enhancement
Converts captured frames for optimal OCR performance.
All operations use OpenCV (no PIL in the hot path).
"""

import cv2
import numpy as np


class Preprocessor:
    """
    Image preprocessing pipeline for OCR enhancement.
    Applies CLAHE, bilateral filtering, and Otsu binarisation.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Shared configuration dict.
        """
        self.config = config

    def process(self, frame_bgra: np.ndarray) -> dict:
        """
        Run the full preprocessing pipeline on a captured frame.

        Args:
            frame_bgra: Raw BGRA frame from mss screen capture.

        Returns:
            dict with keys:
                'original_bgr': Original frame in BGR (for final compositing)
                'gray': Grayscale version
                'enhanced': CLAHE-enhanced grayscale
                'filtered': Bilateral-filtered image
                'binary': Otsu-binarised image (for OCR)
        """
        # Convert BGRA → BGR (drop alpha for compositing)
        original_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # ------------------------------------------------------------------
        # STAGE 2: Convert to grayscale for OCR (keep original RGB for
        #          final compositing)
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------
        # STAGE 2: CLAHE Enhancement
        # Contrast-Limited Adaptive Histogram Equalisation boosts text
        # regions for better OCR accuracy.
        # ------------------------------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # ------------------------------------------------------------------
        # STAGE 2: Bilateral Filtering
        # Reduces noise while preserving text edges. Parameters:
        #   d=9          — diameter of pixel neighbourhood
        #   sigmaColor=75 — filter sigma in the colour space
        #   sigmaSpace=75 — filter sigma in the coordinate space
        # ------------------------------------------------------------------
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # ------------------------------------------------------------------
        # STAGE 2: Adaptive Otsu Binarisation
        # Segments text vs background for cleaner OCR input.
        # ------------------------------------------------------------------
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return {
            "original_bgr": original_bgr,
            "gray": gray,
            "enhanced": enhanced,
            "filtered": filtered,
            "binary": binary,
        }

    def downscale_for_ocr(self, image: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """
        Downscale image for faster OCR processing.

        Args:
            image: Input image (any channel count).
            scale: Scale factor (0.0–1.0). Default 0.5 = half resolution.

        Returns:
            Downscaled image.
        """
        if scale >= 1.0:
            return image.copy()
        h, w = image.shape[:2]
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
