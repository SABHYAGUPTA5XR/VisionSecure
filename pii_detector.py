"""
VisionSecure - STAGE 4: PII Classification (regex + validation hybrid)
Detects all supported PII types from OCR text results.
Each PII type has a dedicated regex pattern and optional validator.

DESIGN NOTE: OCR engines often introduce artifacts - extra spaces, misread
characters (e.g. '@' -> 'a', '.' -> ','), split text across regions. All
patterns are deliberately LENIENT to catch PII even with noisy OCR output.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    pii_type: str           # e.g. 'EMAIL', 'PHONE', 'CREDIT_CARD'
    matched_text: str       # The actual text that matched
    bbox: tuple             # Bounding box (x, y, w, h) from OCR
    confidence: float       # OCR confidence score
    original_text: str      # Full original OCR text from the region
    color: Tuple[int, int, int] = (255, 255, 255)  # BGR display colour


# ------------------------------------------------------------------
# STAGE 4: Colour coding for each PII type (BGR format for OpenCV)
# ------------------------------------------------------------------
PII_COLORS = {
    "EMAIL":         (0, 0, 255),       # RED
    "PHONE":         (0, 140, 255),     # ORANGE
    "IP_ADDRESS":    (0, 255, 255),     # YELLOW
    "CREDIT_CARD":   (200, 0, 180),     # PURPLE
    "CVV":           (200, 200, 200),   # WHITE-ISH
    "API_KEY":       (255, 100, 0),     # BLUE
    "AADHAAR":       (255, 255, 0),     # CYAN
    "PAN_CARD":      (180, 0, 255),     # PINK
    "PASSWORD":      (200, 200, 200),   # WHITE
    "CUSTOM":        (0, 255, 0),       # GREEN
}


class PIIDetector:
    """
    STAGE 4: PII Classification engine.
    Uses LENIENT regex patterns to handle noisy OCR output.
    Validators (Luhn, IP range) run after regex to reduce false positives.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Shared configuration dict containing PII toggle states.
        """
        self.config = config

        # ------------------------------------------------------------------
        # STAGE 4: Regex patterns for each PII type
        # All patterns are deliberately lenient to handle OCR noise.
        # ------------------------------------------------------------------
        self._patterns = {
            # EMAIL: Simple pattern — user@domain.tld
            # Catches: user@gmail.com, name.surname@vitstudent.ac.in
            "EMAIL": re.compile(
                r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
                re.IGNORECASE
            ),

            # PHONE: Global approach - any sequence of 10+ digits.
            # Allows optional country code (+91, +1, etc), dashes, spaces.
            # Will match: 9876543210, +91-9876543210, 98765 43210, etc.
            "PHONE": re.compile(
                r'(?:\+?\d{1,3}[\s\-.]?)?'    # optional country code
                r'(?:\(?\d{2,5}\)?[\s\-.]?)?'  # optional area code
                r'\d[\d\s\-.]{8,}\d',          # core: 10+ digits with separators
                re.IGNORECASE
            ),

            # IP_ADDRESS: Allow OCR to misread dots as commas or spaces.
            # Standard: 192.168.1.100, also: 192 .168.1.100, 192. 168.1.100
            # OCR can produce multi-char separators like ". " or " ."
            "IP_ADDRESS": re.compile(
                r'\d{1,3}'
                r'\s*[.,]\s*'                  # dot/comma with optional spaces
                r'\d{1,3}'
                r'\s*[.,]\s*'
                r'\d{1,3}'
                r'\s*[.,]\s*'
                r'\d{1,3}'
            ),

            # CREDIT_CARD: 13-19 digits with optional separators.
            "CREDIT_CARD": re.compile(
                r'\d{4}[\s\-.]?\d{4}[\s\-.]?\d{4}[\s\-.]?\d{1,4}'
            ),

            # CVV: 3-4 digit codes near card-related keywords.
            "CVV": re.compile(
                r'(?:cvv|cvc|csv|security\s*code|sec\s*code)'
                r'[\s:=]*'
                r'(\d{3,4})',
                re.IGNORECASE
            ),

            # API_KEY: Long alphanumeric strings. Lowered threshold to 16
            # chars and added more prefix patterns.
            "API_KEY": re.compile(
                r'(?:'
                r'(?:sk|pk|api|key|token|bearer|secret|access)'  # keyword prefix
                r'[\s:=_\-]*'
                r')'
                r'[A-Za-z0-9_\-]{16,}',       # 16+ alphanumeric chars
                re.IGNORECASE
            ),

            # AADHAAR: 12 digits, with or without separators.
            # Matches: 2345 6789 0123, 234567890123, 2345-6789-0123
            "AADHAAR": re.compile(
                r'[2-9]\d{3}'                  # first 4 digits (can't start with 0-1)
                r'[\s\-.]?'                    # optional separator
                r'\d{4}'                       # middle 4 digits
                r'[\s\-.]?'                    # optional separator
                r'\d{4}'                       # last 4 digits
            ),

            # PAN_CARD: ABCDE1234F format. Allow lowercase from OCR.
            # OCR may insert spaces between the letter/digit groups.
            "PAN_CARD": re.compile(
                r'[A-Za-z]{5}\s*[0-9]{4}\s*[A-Za-z]'
            ),

            # PASSWORD: Text following password-like keywords.
            # Very lenient: keyword, then optional colon/equals/space, then value.
            # Matches: password: abc123, Password=secret, pwd abc, pass:xyz
            "PASSWORD": re.compile(
                r'(?:password|passwd|pwd|pass|passcode|passphrase)'
                r'[\s:=\->\|]*'               # separator (colon, equals, arrow, etc.)
                r'(\S+)',                       # the password value
                re.IGNORECASE
            ),
        }

        # Custom patterns added by user at runtime
        self._custom_patterns: dict = {}  # label -> compiled regex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_custom_pattern(self, label: str, pattern_str: str):
        """Add a user-defined regex pattern."""
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
            self._custom_patterns[label] = compiled
        except re.error:
            pass  # Ignore invalid regex from user input

    def remove_custom_pattern(self, label: str):
        """Remove a user-defined regex pattern."""
        self._custom_patterns.pop(label, None)

    def detect(
        self, ocr_results: List[Tuple[str, tuple, float]]
    ) -> List[PIIMatch]:
        """
        STAGE 4: Scan OCR results for PII matches.

        Args:
            ocr_results: List of (text, bbox, confidence) from OCR engine.

        Returns:
            List of PIIMatch objects for all detected PII.
        """
        matches = []
        pii_toggles = self.config.get("pii_toggles", {})

        # Track which bboxes already have a built-in PII match to
        # avoid double-redaction when a custom regex also matches.
        matched_bboxes = set()

        for text, bbox, conf in ocr_results:
            if not text or len(text.strip()) < 2:
                continue

            # Normalise OCR text: strip extra whitespace but keep internal spaces
            clean_text = text.strip()

            # Track if this OCR region already matched a built-in PII type
            region_matched = False

            # Check each built-in PII type
            for pii_type, pattern in self._patterns.items():
                # Skip disabled PII types
                # Support both old "INDIAN_MOBILE" and new "PHONE" toggle keys
                if pii_type == "PHONE":
                    if not pii_toggles.get("PHONE", pii_toggles.get("INDIAN_MOBILE", True)):
                        continue
                elif not pii_toggles.get(pii_type, True):
                    continue

                for m in pattern.finditer(clean_text):
                    matched_text = m.group(0)

                    # STAGE 4: Post-match validation to reduce false positives

                    # Luhn validation for credit card numbers
                    if pii_type == "CREDIT_CARD":
                        if not self._luhn_check(matched_text):
                            continue

                    # IP address: validate octets are 0-255
                    if pii_type == "IP_ADDRESS":
                        if not self._validate_ip(matched_text):
                            continue

                    # PHONE: ensure at least 10 actual digits
                    if pii_type == "PHONE":
                        digit_count = sum(1 for c in matched_text if c.isdigit())
                        if digit_count < 10:
                            continue

                    # AADHAAR: must have exactly 12 digits
                    if pii_type == "AADHAAR":
                        digit_count = sum(1 for c in matched_text if c.isdigit())
                        if digit_count != 12:
                            continue

                    # API_KEY: skip if it's too short or just a common word
                    if pii_type == "API_KEY":
                        alnum_part = re.sub(r'[\s:=_\-]', '', matched_text)
                        if len(alnum_part) < 16:
                            continue

                    # For PASSWORD and CVV, extract the captured group
                    if pii_type in ("PASSWORD", "CVV") and m.lastindex:
                        matched_text = m.group(1)

                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        matched_text=matched_text,
                        bbox=bbox,
                        confidence=conf,
                        original_text=clean_text,
                        color=PII_COLORS.get(pii_type, (255, 255, 255)),
                    ))
                    region_matched = True

            # STAGE 4: Check custom patterns — but SKIP if this
            # OCR region already matched a built-in PII type, to
            # avoid double-redacting the same bounding box.
            if not region_matched and pii_toggles.get("CUSTOM", True):
                for label, pattern in self._custom_patterns.items():
                    for m in pattern.finditer(clean_text):
                        matches.append(PIIMatch(
                            pii_type="CUSTOM",
                            matched_text=m.group(0),
                            bbox=bbox,
                            confidence=conf,
                            original_text=clean_text,
                            color=PII_COLORS["CUSTOM"],
                        ))

        return matches

    # ------------------------------------------------------------------
    # STAGE 4: Validators
    # ------------------------------------------------------------------

    @staticmethod
    def _luhn_check(card_str: str) -> bool:
        """
        Validate a credit card number using the Luhn algorithm.
        Returns True if the number passes the checksum.
        """
        digits = [int(d) for d in card_str if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False

        reversed_digits = digits[::-1]
        total = 0
        for i, d in enumerate(reversed_digits):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    @staticmethod
    def _validate_ip(ip_str: str) -> bool:
        """
        Validate that each octet of an IPv4 address is 0-255.
        Handles OCR artifacts (commas, spaces) as dot substitutes.
        """
        try:
            # Split on dots, commas, spaces, or combinations thereof
            parts = re.split(r'[.,\s]+', ip_str.strip())
            # Filter out empty strings from split
            parts = [p for p in parts if p]
            if len(parts) != 4:
                return False
            for p in parts:
                if not p.isdigit():
                    return False
                val = int(p)
                if val < 0 or val > 255:
                    return False
            return True
        except (ValueError, AttributeError):
            return False
