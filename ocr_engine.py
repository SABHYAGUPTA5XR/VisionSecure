"""
VisionSecure — STAGE 3: Text Detection & OCR
Wraps EasyOCR for text detection and recognition.
Implements frame differencing cache to avoid redundant OCR on static regions.
"""

import time
import cv2
import numpy as np
from typing import List, Tuple, Optional


class OCREngine:
    """
    STAGE 3: Text detection and OCR using EasyOCR.
    Initialises the reader once and reuses it for all frames.
    Includes frame caching to skip OCR on unchanged regions.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Shared configuration dict.
        """
        self.config = config
        self._reader = None
        self._last_frame_hash = None
        self._cached_results: List[Tuple[str, tuple, float]] = []
        self._last_ocr_time = 0.0
        self._initialised = False

    # ------------------------------------------------------------------
    # STAGE 3: Lazy initialisation (EasyOCR is heavy on first load)
    # ------------------------------------------------------------------

    def initialise(self):
        """
        Initialise EasyOCR reader. Call this once before first use.
        Separated from __init__ so the GUI can show a loading indicator.
        """
        if self._initialised:
            return
        try:
            import easyocr
            gpu_available = False
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                pass

            self._reader = easyocr.Reader(
                ["en"],
                gpu=gpu_available,
                verbose=False,
            )
            self._initialised = True
        except Exception as e:
            print(f"[OCR Engine] Failed to initialise EasyOCR: {e}")
            self._initialised = False

    @property
    def is_ready(self):
        return self._initialised and self._reader is not None

    # ------------------------------------------------------------------
    # STAGE 3: Run OCR on a preprocessed frame
    # ------------------------------------------------------------------

    def detect_text(
        self, image: np.ndarray, original_shape: Tuple[int, int] = None,
        scale: float = 1.0
    ) -> List[Tuple[str, tuple, float]]:
        """
        STAGE 3: Detect and recognise text in the image.

        Args:
            image: Preprocessed grayscale or BGR image.
            original_shape: (height, width) of the original frame for bbox scaling.
            scale: Scale factor used during downscaling (to map bboxes back).

        Returns:
            List of (text_string, bounding_box, confidence_score) tuples.
            bounding_box is (x, y, w, h) in original frame coordinates.
        """
        if not self.is_ready:
            return []

        # ------------------------------------------------------------------
        # STAGE 3: Frame differencing cache
        # If the frame hasn't changed significantly, return cached results.
        # ------------------------------------------------------------------
        frame_hash = self._compute_frame_hash(image)
        if frame_hash == self._last_frame_hash and self._cached_results:
            return self._cached_results

        confidence_threshold = self.config.get("confidence_threshold", 60) / 100.0

        try:
            ocr_start = time.perf_counter()

            # STAGE 3: Run EasyOCR (psm=6 equivalent via paragraph=False)
            raw_results = self._reader.readtext(
                image,
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
            )

            self._last_ocr_time = time.perf_counter() - ocr_start

        except Exception as e:
            print(f"[OCR Engine] Error during OCR: {e}")
            return self._cached_results if self._cached_results else []

        results = []

        for detection in raw_results:
            bbox_points, text, conf = detection

            # STAGE 3: Confidence thresholding (>= threshold)
            if conf < confidence_threshold:
                continue

            # Convert EasyOCR bbox format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            # to (x, y, w, h) format
            points = np.array(bbox_points, dtype=np.float32)
            x_min = int(points[:, 0].min())
            y_min = int(points[:, 1].min())
            x_max = int(points[:, 0].max())
            y_max = int(points[:, 1].max())

            # Scale bounding box back to original frame coordinates
            if scale != 1.0 and scale > 0:
                inv_scale = 1.0 / scale
                x_min = int(x_min * inv_scale)
                y_min = int(y_min * inv_scale)
                x_max = int(x_max * inv_scale)
                y_max = int(y_max * inv_scale)

            w = x_max - x_min
            h = y_max - y_min

            if w > 0 and h > 0:
                results.append((text.strip(), (x_min, y_min, w, h), conf))

        # Update cache
        self._last_frame_hash = frame_hash
        self._cached_results = results

        return results

    @property
    def last_ocr_time_ms(self) -> float:
        """Return the last OCR processing time in milliseconds."""
        return self._last_ocr_time * 1000.0

    # ------------------------------------------------------------------
    # Frame differencing for cache
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_frame_hash(image: np.ndarray) -> int:
        """
        Compute a fast perceptual hash of the frame.
        Uses a downscaled version to detect significant changes.
        """
        small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Quantise to 4-bit to ignore minor noise
        quantised = (small // 16).astype(np.uint8)
        return hash(quantised.tobytes())
