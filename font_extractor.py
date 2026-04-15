"""
VisionSecure — Font Metric Extraction
Analyses a region of interest (ROI) to estimate font properties
for pixel-perfect Ghost Mask text rendering.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FontMetrics:
    """Estimated font properties from an image ROI."""
    font_size: int              # Approximate font size in pixels
    text_color: Tuple[int, int, int]  # BGR text colour
    bg_color: Tuple[int, int, int]    # BGR background colour
    is_dark_bg: bool            # True if background is dark (light text on dark)
    line_height: int            # Pixel height of the text line
    char_width: int             # Estimated average character width


class FontExtractor:
    """
    Analyses image ROIs to estimate font metrics for Ghost Mask rendering.
    Uses colour histograms and spatial analysis to determine text/background
    colours and approximate font dimensions.
    """

    def extract(self, roi_bgr: np.ndarray, text_length: int = 10) -> FontMetrics:
        """
        Extract font metrics from an ROI containing text.

        Args:
            roi_bgr: BGR image region containing the text.
            text_length: Number of characters in the detected text (for width estimation).

        Returns:
            FontMetrics with estimated properties.
        """
        h, w = roi_bgr.shape[:2]

        # ------------------------------------------------------------------
        # Determine background vs text colour using K-means clustering.
        # The dominant colour is background; the secondary is text.
        # ------------------------------------------------------------------
        bg_color, text_color = self._extract_colors(roi_bgr)

        # Determine if background is dark
        bg_luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
        is_dark_bg = bg_luminance < 128

        # ------------------------------------------------------------------
        # Estimate font size from ROI height.
        # Typical text occupies ~70-80% of the bounding box height.
        # ------------------------------------------------------------------
        font_size = max(int(h * 0.75), 8)
        line_height = h

        # Estimate character width from ROI width and text length
        char_width = max(w // max(text_length, 1), 4)

        return FontMetrics(
            font_size=font_size,
            text_color=tuple(int(c) for c in text_color),
            bg_color=tuple(int(c) for c in bg_color),
            is_dark_bg=is_dark_bg,
            line_height=line_height,
            char_width=char_width,
        )

    def _extract_colors(
        self, roi_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the two dominant colours from the ROI using K-means.
        Returns (background_color, text_color) in BGR.
        """
        h, w = roi_bgr.shape[:2]

        # Flatten pixels for clustering
        pixels = roi_bgr.reshape(-1, 3).astype(np.float32)

        if len(pixels) < 2:
            # Fallback for tiny ROIs
            mean_color = pixels.mean(axis=0) if len(pixels) > 0 else np.array([128, 128, 128])
            return mean_color, np.array([255, 255, 255]) - mean_color

        # K-means with k=2 (text + background)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
        except cv2.error:
            # Fallback
            mean_color = pixels.mean(axis=0)
            return mean_color, np.array([255, 255, 255]) - mean_color

        # The cluster with more pixels is the background
        label_counts = np.bincount(labels.flatten())
        bg_label = np.argmax(label_counts)
        text_label = 1 - bg_label

        bg_color = centers[bg_label]
        text_color = centers[text_label]

        return bg_color, text_color
