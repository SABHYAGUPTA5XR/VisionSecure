"""
VisionSecure — STAGE 5: Redaction Engine
Two modes: Gaussian Blur and Ghost Mask (synthetic text replacement).
All image compositing uses OpenCV; Ghost Mask text rendering uses Pillow.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from font_extractor import FontExtractor, FontMetrics
from pii_detector import PIIMatch


class Redactor:
    """
    STAGE 5: Redaction engine supporting Gaussian Blur and Ghost Mask.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Shared configuration dict with redaction settings.
        """
        self.config = config
        self.font_extractor = FontExtractor()

        # Session statistics
        self.frame_redaction_count = 0
        self.session_redaction_count = 0

        # Try to load a monospace font for Ghost Mask
        self._pil_font_cache = {}

    # ------------------------------------------------------------------
    # STAGE 5: Main redaction entry point
    # ------------------------------------------------------------------

    def redact_frame(
        self, frame_bgr: np.ndarray, pii_matches: List[PIIMatch]
    ) -> np.ndarray:
        """
        Apply redaction to all detected PII in the frame.

        Args:
            frame_bgr: Original BGR frame.
            pii_matches: List of detected PII matches with bounding boxes.

        Returns:
            Redacted BGR frame.
        """
        result = frame_bgr.copy()
        self.frame_redaction_count = 0

        redaction_modes = self.config.get("redaction_modes", {})

        for match in pii_matches:
            mode = redaction_modes.get(match.pii_type, "blur")

            x, y, w, h = match.bbox
            # Clamp to frame bounds
            fh, fw = result.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, fw - x)
            h = min(h, fh - y)

            if w <= 0 or h <= 0:
                continue

            if mode == "ghost_mask":
                result = self._apply_ghost_mask(result, match, x, y, w, h)
            else:
                result = self._apply_gaussian_blur(result, x, y, w, h)

            self.frame_redaction_count += 1
            self.session_redaction_count += 1

        return result

    # ------------------------------------------------------------------
    # STAGE 5 — MODE A: Gaussian Blur
    # ------------------------------------------------------------------

    def _apply_gaussian_blur(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        STAGE 5 MODE A: Apply Gaussian blur with feathered edges.

        Steps:
          1. Extract the ROI bounding box from the original colour frame.
          2. Apply Gaussian blur with configurable kernel size and sigma.
          3. Feather the blur mask edges using a 5px Gaussian gradient.
          4. Composite blurred patch back onto the frame seamlessly.
        """
        kernel_size = self.config.get("blur_kernel_size", 51)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = self.config.get("blur_sigma", 15)

        # Extract ROI
        roi = frame[y:y+h, x:x+w].copy()

        # Apply heavy Gaussian blur
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigma)

        # ------------------------------------------------------------------
        # STAGE 5: Feather the blur mask edges using a 5px Gaussian gradient
        # at ROI boundary for seamless compositing.
        # ------------------------------------------------------------------
        feather_px = 5
        mask = np.ones((h, w), dtype=np.float32)

        # Create gradient at edges
        for i in range(feather_px):
            alpha = (i + 1) / feather_px
            if i < h:
                mask[i, :] = min(mask[i, 0], alpha)
                mask[h - 1 - i, :] = min(mask[h - 1 - i, 0], alpha)
            if i < w:
                mask[:, i] = np.minimum(mask[:, i], alpha)
                mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

        # Blend blurred and original using the feather mask
        mask_3ch = np.stack([mask] * 3, axis=-1)
        composited = (blurred * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

        frame[y:y+h, x:x+w] = composited
        return frame

    # ------------------------------------------------------------------
    # STAGE 5 — MODE B: Ghost Mask (primary innovation)
    # ------------------------------------------------------------------

    def _apply_ghost_mask(
        self, frame: np.ndarray, match: PIIMatch,
        x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        STAGE 5 MODE B: Ghost Mask — synthetic text replacement.

        Steps:
          1. Classify PII type.
          2. Generate contextually plausible synthetic replacement.
          3. Extract font metrics from ROI via image analysis.
          4. Render synthetic text using extracted font metrics via Pillow.
          5. Pixel-perfect composite the replacement into the original frame.
        """
        # Extract ROI for font analysis
        roi = frame[y:y+h, x:x+w].copy()

        # Step 1-2: Generate synthetic replacement text
        synthetic_text = self._generate_synthetic(match.pii_type, match.matched_text)

        # Step 3: Extract font metrics from ROI
        metrics = self.font_extractor.extract(roi, len(match.matched_text))

        # Step 4: Render synthetic text using Pillow
        rendered = self._render_text(synthetic_text, w, h, metrics)

        # Step 5: Composite into frame
        if rendered is not None:
            frame[y:y+h, x:x+w] = rendered
        else:
            # Fallback to blur if rendering fails
            frame = self._apply_gaussian_blur(frame, x, y, w, h)

        return frame

    # ------------------------------------------------------------------
    # STAGE 5: Ghost Mask — synthetic text generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_synthetic(pii_type: str, original: str) -> str:
        """
        Generate a contextually plausible synthetic replacement.
        Masks sensitive digits/characters while preserving format.
        """
        if pii_type == "EMAIL":
            # p**********@gmail.com (mask local-part after first char)
            parts = original.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                if len(local) > 1:
                    return local[0] + "*" * (len(local) - 1) + "@" + domain
                return "*@" + domain
            return "*" * len(original)

        elif pii_type == "PHONE":
            # 98*****210 (mask middle digits)
            digits = "".join(c for c in original if c.isdigit())
            if len(digits) >= 10:
                return digits[:2] + "*" * (len(digits) - 5) + digits[-3:]
            return "*" * len(original)

        elif pii_type == "IP_ADDRESS":
            # 192.168.***.*** (mask last two octets)
            octets = original.split(".")
            if len(octets) == 4:
                return octets[0] + "." + octets[1] + ".***" + ".***"
            return "*" * len(original)

        elif pii_type == "CREDIT_CARD":
            # **** **** **** 4242 (mask all but last 4)
            digits = "".join(c for c in original if c.isdigit())
            if len(digits) >= 4:
                return "**** **** **** " + digits[-4:]
            return "*" * len(original)

        elif pii_type == "API_KEY":
            # sk-********************XyZ9 (show first 3 + last 4)
            if len(original) > 7:
                return original[:3] + "*" * (len(original) - 7) + original[-4:]
            return "*" * len(original)

        elif pii_type == "AADHAAR":
            # XXXX XXXX 6789
            digits = "".join(c for c in original if c.isdigit())
            if len(digits) >= 4:
                return "XXXX XXXX " + digits[-4:]
            return "XXXX XXXX XXXX"

        elif pii_type == "PAN_CARD":
            # Mask middle portion: A****1234Z → A****234Z
            if len(original) >= 10:
                return original[0] + "****" + original[5:]
            return "*" * len(original)

        elif pii_type == "PASSWORD":
            return "*" * max(len(original), 8)

        elif pii_type == "CVV":
            return "***"

        else:
            # CUSTOM / unknown
            return "*" * len(original)

    # ------------------------------------------------------------------
    # STAGE 5: Ghost Mask — text rendering via Pillow
    # ------------------------------------------------------------------

    def _render_text(
        self, text: str, width: int, height: int, metrics: FontMetrics
    ) -> Optional[np.ndarray]:
        """
        Render synthetic text onto a background matching the ROI.
        Uses Pillow for high-quality text rendering, then converts to OpenCV.
        """
        try:
            # Create background filled with extracted bg colour (RGB for Pillow)
            bg_rgb = (metrics.bg_color[2], metrics.bg_color[1], metrics.bg_color[0])
            text_rgb = (metrics.text_color[2], metrics.text_color[1], metrics.text_color[0])

            img = Image.new("RGB", (width, height), bg_rgb)
            draw = ImageDraw.Draw(img)

            # Get or create font at the estimated size
            font = self._get_font(metrics.font_size)

            # Calculate text position (vertically centred)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # If text is wider than ROI, reduce font size
            if text_w > width and width > 0:
                scale_factor = width / text_w * 0.9
                adjusted_size = max(int(metrics.font_size * scale_factor), 6)
                font = self._get_font(adjusted_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

            # Centre the text in the ROI
            tx = max((width - text_w) // 2, 2)
            ty = max((height - text_h) // 2, 0)

            draw.text((tx, ty), text, fill=text_rgb, font=font)

            # Convert Pillow → OpenCV (RGB → BGR)
            result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return result

        except Exception as e:
            print(f"[Redactor] Ghost mask rendering error: {e}")
            return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a cached Pillow font at the given size."""
        if size in self._pil_font_cache:
            return self._pil_font_cache[size]

        try:
            # Try common monospace fonts
            for font_name in ["consola.ttf", "cour.ttf", "arial.ttf", "DejaVuSansMono.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, size)
                    self._pil_font_cache[size] = font
                    return font
                except (OSError, IOError):
                    continue

            # Fallback to default
            font = ImageFont.load_default()
            self._pil_font_cache[size] = font
            return font
        except Exception:
            font = ImageFont.load_default()
            self._pil_font_cache[size] = font
            return font
