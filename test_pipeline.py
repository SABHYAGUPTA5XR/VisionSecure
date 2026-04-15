"""
VisionSecure - Pipeline Validation Test
Tests all stages individually with synthetic data.
Tests the LENIENT regex patterns against realistic OCR output.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_preprocessor():
    """Test STAGE 2: Preprocessing pipeline."""
    from preprocess import Preprocessor

    config = {}
    preprocessor = Preprocessor(config)

    # Create a synthetic BGRA frame (like mss output)
    frame = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
    result = preprocessor.process(frame)

    assert "original_bgr" in result, "Missing original_bgr"
    assert "gray" in result, "Missing gray"
    assert "enhanced" in result, "Missing enhanced (CLAHE)"
    assert "filtered" in result, "Missing filtered (bilateral)"
    assert "binary" in result, "Missing binary (Otsu)"
    assert result["original_bgr"].shape == (480, 640, 3), "BGR shape wrong"
    assert result["gray"].shape == (480, 640), "Gray shape wrong"
    assert result["binary"].shape == (480, 640), "Binary shape wrong"

    # Test downscaling
    downscaled = preprocessor.downscale_for_ocr(result["enhanced"], 0.5)
    assert downscaled.shape == (240, 320), f"Downscale wrong: {downscaled.shape}"

    print("[PASS] STAGE 2: Preprocessor - ALL TESTS PASSED")


def test_pii_detector():
    """Test STAGE 4: PII Detection with known patterns - LENIENT mode."""
    from pii_detector import PIIDetector

    config = {"pii_toggles": {k: True for k in [
        "EMAIL", "PHONE", "IP_ADDRESS", "CREDIT_CARD",
        "CVV", "API_KEY", "AADHAAR", "PAN_CARD", "PASSWORD", "CUSTOM"
    ]}}
    detector = PIIDetector(config)

    bbox_dummy = (0, 0, 100, 20)

    # ---- EMAIL ----
    email_tests = [
        "Email: john.doe@example.com",
        "john.doe@example.com",
        "contact user@domain.co",
        "a@b.cd",
    ]
    for text in email_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "EMAIL" in found_types, f"EMAIL not found in: '{text}' (found: {found_types})"
    print("  [OK] EMAIL detection works")

    # ---- PHONE (10+ digits, global) ----
    phone_tests = [
        "Call 9876543210",
        "+91-9876543210",
        "Phone: 98765 43210",
        "1234567890",
        "+1-555-123-4567",
    ]
    for text in phone_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "PHONE" in found_types, f"PHONE not found in: '{text}' (found: {found_types})"
    print("  [OK] PHONE detection works (global 10+ digits)")

    # ---- IP ADDRESS ----
    ip_tests = [
        "Server IP: 192.168.1.100",
        "192.168.1.100",
        "IP: 10.0.0.1",
    ]
    for text in ip_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "IP_ADDRESS" in found_types, f"IP not found in: '{text}' (found: {found_types})"
    print("  [OK] IP_ADDRESS detection works")

    # ---- CREDIT CARD (Luhn valid) ----
    card_tests = [
        "Card: 4532015112830366",
        "4532 0151 1283 0366",
    ]
    for text in card_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "CREDIT_CARD" in found_types, f"CARD not found in: '{text}' (found: {found_types})"
    # Invalid Luhn should be rejected
    invalid_card = [("Card: 1234567890123456", bbox_dummy, 0.9)]
    matches = detector.detect(invalid_card)
    card_matches = [m for m in matches if m.pii_type == "CREDIT_CARD"]
    assert len(card_matches) == 0, "Should reject invalid Luhn"
    print("  [OK] CREDIT_CARD detection works (with Luhn)")

    # ---- AADHAAR (12 digits, with or without separators) ----
    aadhaar_tests = [
        "Aadhaar: 2345 6789 0123",
        "2345-6789-0123",
        "234567890123",
    ]
    for text in aadhaar_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "AADHAAR" in found_types, f"AADHAAR not found in: '{text}' (found: {found_types})"
    print("  [OK] AADHAAR detection works")

    # ---- PAN CARD ----
    pan_tests = [
        "PAN: ABCDE1234F",
        "ABCDE1234F",
    ]
    for text in pan_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "PAN_CARD" in found_types, f"PAN not found in: '{text}' (found: {found_types})"
    print("  [OK] PAN_CARD detection works")

    # ---- PASSWORD ----
    password_tests = [
        "password: MyS3cretPass!",
        "Password=hello123",
        "pwd abc123",
        "pass: test",
    ]
    for text in password_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "PASSWORD" in found_types, f"PASSWORD not found in: '{text}' (found: {found_types})"
    print("  [OK] PASSWORD detection works")

    # ---- API KEY ----
    api_key_tests = [
        "API Key: sk-abcdef1234567890abcdefghij",
        "token: abcdef1234567890abcd",
        "key=ABCDEF1234567890GHIJ",
        "secret: xyzxyzxyzxyzxyzxyz12",
    ]
    for text in api_key_tests:
        matches = detector.detect([(text, bbox_dummy, 0.9)])
        found_types = [m.pii_type for m in matches]
        assert "API_KEY" in found_types, f"API_KEY not found in: '{text}' (found: {found_types})"
    print("  [OK] API_KEY detection works")

    # ---- CUSTOM PATTERNS ----
    detector.add_custom_pattern("SSN", r"\d{3}-\d{2}-\d{4}")
    custom_test = [("SSN: 123-45-6789", bbox_dummy, 0.9)]
    matches = detector.detect(custom_test)
    custom_matches = [m for m in matches if m.pii_type == "CUSTOM"]
    assert len(custom_matches) > 0, "Custom pattern not detected"
    print("  [OK] CUSTOM pattern detection works")

    # ---- IP VALIDATION (invalid octets) ----
    invalid_ip = [("IP: 999.999.999.999", bbox_dummy, 0.9)]
    matches = detector.detect(invalid_ip)
    ip_matches = [m for m in matches if m.pii_type == "IP_ADDRESS"]
    assert len(ip_matches) == 0, "Should reject invalid IP octets"
    print("  [OK] IP validation rejects invalid octets")

    print("[PASS] STAGE 4: PII Detector - ALL TESTS PASSED")


def test_font_extractor():
    """Test font metric extraction."""
    from font_extractor import FontExtractor

    fe = FontExtractor()

    # Dark background with light text
    dark_roi = np.zeros((30, 200, 3), dtype=np.uint8)
    dark_roi[:] = (30, 30, 30)
    cv2.putText(dark_roi, "test@email.com", (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    metrics = fe.extract(dark_roi, text_length=14)
    assert metrics.is_dark_bg, "Should detect dark background"
    assert metrics.font_size > 0, "Font size should be positive"

    # Light background with dark text
    light_roi = np.ones((30, 200, 3), dtype=np.uint8) * 240
    cv2.putText(light_roi, "some text", (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)

    metrics = fe.extract(light_roi, text_length=9)
    assert not metrics.is_dark_bg, "Should detect light background"

    print("[PASS] Font Extractor - ALL TESTS PASSED")


def test_redactor():
    """Test STAGE 5: Redaction - both blur and ghost mask modes."""
    from redactor import Redactor
    from pii_detector import PIIMatch

    config = {
        "blur_kernel_size": 51,
        "blur_sigma": 15,
        "redaction_modes": {"EMAIL": "blur", "PHONE": "ghost_mask"},
    }
    redactor = Redactor(config)

    # Create a test frame with "text" on it
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.putText(frame, "john@example.com", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, "9876543210", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    pii_matches = [
        PIIMatch(pii_type="EMAIL", matched_text="john@example.com",
                 bbox=(50, 75, 280, 35), confidence=0.95,
                 original_text="john@example.com"),
        PIIMatch(pii_type="PHONE", matched_text="9876543210",
                 bbox=(50, 175, 240, 35), confidence=0.92,
                 original_text="9876543210"),
    ]

    redacted = redactor.redact_frame(frame, pii_matches)

    assert redacted.shape == frame.shape, "Redacted frame shape mismatch"
    assert redactor.frame_redaction_count == 2, "Should have 2 redactions"
    assert redactor.session_redaction_count == 2, "Session total wrong"

    print("[PASS] STAGE 5: Redactor - ALL TESTS PASSED")


def test_synthetic_text_generation():
    """Test Ghost Mask synthetic text generation for ALL PII types."""
    from redactor import Redactor

    tests = [
        ("EMAIL", "john.doe@gmail.com", lambda r: "@gmail.com" in r and r.startswith("j")),
        ("PHONE", "9876543210", lambda r: "*" in r),
        ("IP_ADDRESS", "192.168.1.100", lambda r: r.startswith("192.168") and "***" in r),
        ("CREDIT_CARD", "4532 0151 1283 0366", lambda r: r.endswith("0366") and "****" in r),
        ("AADHAAR", "2345 6789 0123", lambda r: "XXXX" in r and "0123" in r),
        ("API_KEY", "sk-abcdef1234567890abcdefgh", lambda r: r.startswith("sk-") and "*" in r),
        ("PAN_CARD", "ABCDE1234F", lambda r: r.startswith("A") and "****" in r),
        ("PASSWORD", "MyS3cretPass!", lambda r: "*" in r and len(r) >= 8),
        ("CVV", "123", lambda r: r == "***"),
        ("CUSTOM", "anything", lambda r: "*" in r),
    ]

    for pii_type, original, check in tests:
        result = Redactor._generate_synthetic(pii_type, original)
        assert check(result), f"{pii_type}: mask failed for '{original}' -> '{result}'"

    print("[PASS] STAGE 5: Ghost Mask Synthetic Text - ALL TESTS PASSED")


def test_capture():
    """Test STAGE 1: Screen capture basics."""
    import queue
    from capture import ScreenCapture

    config = {"target_fps": 10, "monitor_index": 1}
    fq = queue.Queue(maxsize=2)
    cap = ScreenCapture(config, fq)

    cap.start()
    assert cap.is_running, "Should be running after start"

    import time
    time.sleep(0.5)

    assert not fq.empty(), "Should have captured at least one frame"

    frame = fq.get_nowait()
    assert isinstance(frame, np.ndarray), "Frame should be numpy array"
    assert frame.ndim == 3, "Frame should be 3D (H, W, C)"
    assert frame.shape[2] == 4, "Frame should be BGRA (4 channels)"

    cap.stop()
    assert not cap.is_running, "Should not be running after stop"

    print("[PASS] STAGE 1: Screen Capture - ALL TESTS PASSED")


def test_output_window_hud():
    """Test STAGE 6: HUD rendering on a synthetic frame."""
    from output_window import OutputWindow
    import queue

    config = {"show_bounding_boxes": True}
    oq = queue.Queue(maxsize=2)
    ow = OutputWindow(config, oq)

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    stats = {"fps": 24.5, "frame_redactions": 3, "session_total": 87}

    hud_frame = ow._draw_hud(frame, stats)
    assert hud_frame.shape == frame.shape, "HUD should preserve frame shape"

    from pii_detector import PIIMatch
    matches = [
        PIIMatch("EMAIL", "test@test.com", (10, 10, 200, 30), 0.9, "test@test.com"),
    ]
    annotated = ow._draw_annotations(frame.copy(), matches)
    assert annotated.shape == frame.shape, "Annotations should preserve frame shape"

    print("[PASS] STAGE 6: Output Window HUD - ALL TESTS PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  VisionSecure - Pipeline Validation Tests")
    print("=" * 60)
    print()

    tests = [
        ("Preprocessor (STAGE 2)", test_preprocessor),
        ("PII Detector (STAGE 4)", test_pii_detector),
        ("Font Extractor", test_font_extractor),
        ("Redactor (STAGE 5)", test_redactor),
        ("Synthetic Text", test_synthetic_text_generation),
        ("Screen Capture (STAGE 1)", test_capture),
        ("Output HUD (STAGE 6)", test_output_window_hud),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
