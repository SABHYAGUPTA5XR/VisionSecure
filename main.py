"""
VisionSecure — Main Entry Point
Launches the full pipeline: screen capture → preprocess → OCR → PII detect →
redact → output display, plus the tkinter control panel GUI.

Architecture:
  - Main thread: tkinter control panel
  - Thread 1: Screen capture (mss)
  - Thread 2: Processing pipeline (preprocess → OCR → PII → redact)
  - Thread 3: Output window (OpenCV imshow + HUD)
"""

import sys
import os
import time
import threading
import queue
import tkinter as tk
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capture import ScreenCapture
from preprocess import Preprocessor
from ocr_engine import OCREngine
from pii_detector import PIIDetector
from redactor import Redactor
from output_window import OutputWindow
from control_panel import ControlPanel


class VisionSecureApp:
    """
    Main application controller.
    Orchestrates all threads and manages the shared configuration.
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # Shared configuration (read by all threads, written by GUI thread)
        # Thread safety: GUI writes entire dicts atomically; workers read
        # snapshots. No fine-grained locking needed for these value types.
        # ------------------------------------------------------------------
        self.config = {
            "target_fps": 20,
            "monitor_index": 1,
            "confidence_threshold": 40,     # Lowered for better OCR recall
            "blur_kernel_size": 51,
            "blur_sigma": 15,
            "show_bounding_boxes": True,
            "pii_toggles": {
                "EMAIL": True,
                "PHONE": True,
                "IP_ADDRESS": True,
                "CREDIT_CARD": True,
                "CVV": True,
                "API_KEY": True,
                "AADHAAR": True,
                "PAN_CARD": True,
                "PASSWORD": True,
                "CUSTOM": True,
            },
            "redaction_modes": {
                "EMAIL": "blur",
                "PHONE": "blur",
                "IP_ADDRESS": "blur",
                "CREDIT_CARD": "blur",
                "CVV": "blur",
                "API_KEY": "blur",
                "AADHAAR": "blur",
                "PAN_CARD": "blur",
                "PASSWORD": "blur",
                "CUSTOM": "blur",
            },
            "ocr_scale": 0.5,
            # Capture region: default to left half of screen to avoid
            # the "hall of mirrors" recursive capture of the output window.
            # Set to None to capture the full monitor.
            "capture_region": {
                "left": 0, "top": 0, "width": 960, "height": 1080
            },
        }

        # Queues
        self.frame_queue = queue.Queue(maxsize=2)   # capture → processing
        self.output_queue = queue.Queue(maxsize=2)  # processing → display

        # Pipeline components
        self.capture = ScreenCapture(self.config, self.frame_queue)
        self.preprocessor = Preprocessor(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.pii_detector = PIIDetector(self.config)
        self.redactor = Redactor(self.config)
        self.output_window = OutputWindow(self.config, self.output_queue)

        # Processing thread
        self._processing_thread = None
        self._processing_running = threading.Event()

        # Stats
        self._pipeline_fps = 0.0

    # ==================================================================
    # Pipeline lifecycle
    # ==================================================================

    def start_pipeline(self):
        """Start the capture, processing, and output threads."""
        print("[Main] Starting pipeline...")

        # Initialise OCR engine (lazy load)
        if not self.ocr_engine.is_ready:
            print("[Main] Initialising OCR engine (first load may download models)...")
            init_thread = threading.Thread(target=self.ocr_engine.initialise, daemon=True)
            init_thread.start()

        # Start capture
        self.capture.start()

        # Start processing
        self._processing_running.set()
        if self._processing_thread is None or not self._processing_thread.is_alive():
            self._processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self._processing_thread.start()

        # Start output window
        self.output_window.start()

        print("[Main] Pipeline running.")

    def pause_pipeline(self):
        """Toggle pause on the capture thread."""
        if self.capture.is_paused:
            self.capture.resume()
            print("[Main] Pipeline resumed.")
        else:
            self.capture.pause()
            print("[Main] Pipeline paused.")

    def stop_pipeline(self):
        """Stop all threads."""
        print("[Main] Stopping pipeline...")
        self._processing_running.clear()
        self.capture.stop()
        self.output_window.stop()

        # Drain queues
        for q in [self.frame_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        print("[Main] Pipeline stopped.")

    # ==================================================================
    # Processing thread — runs stages 2–5
    # ==================================================================

    def _processing_loop(self):
        """
        Main processing thread. Pulls captured frames and runs the full
        DIP pipeline: preprocess → OCR → PII detect → redact.
        """
        frame_times = []

        while self._processing_running.is_set():
            try:
                frame_bgra = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            process_start = time.perf_counter()

            # ----------------------------------------------------------
            # STAGE 2: Preprocessing
            # ----------------------------------------------------------
            preprocessed = self.preprocessor.process(frame_bgra)
            original_bgr = preprocessed["original_bgr"]

            # ----------------------------------------------------------
            # STAGE 3: Text Detection & OCR
            # ----------------------------------------------------------
            ocr_results = []
            if self.ocr_engine.is_ready:
                # Downscale for OCR performance
                scale = self.config.get("ocr_scale", 0.5)
                ocr_input = self.preprocessor.downscale_for_ocr(
                    preprocessed["enhanced"], scale
                )
                ocr_results = self.ocr_engine.detect_text(
                    ocr_input,
                    original_shape=original_bgr.shape[:2],
                    scale=scale,
                )

            # ----------------------------------------------------------
            # STAGE 4: PII Classification
            # ----------------------------------------------------------
            pii_matches = self.pii_detector.detect(ocr_results)

            # ----------------------------------------------------------
            # STAGE 5: Redaction
            # ----------------------------------------------------------
            redacted = self.redactor.redact_frame(original_bgr, pii_matches)

            # ----------------------------------------------------------
            # Push to output queue
            # ----------------------------------------------------------
            stats = {
                "fps": self._pipeline_fps,
                "frame_redactions": self.redactor.frame_redaction_count,
                "session_total": self.redactor.session_redaction_count,
            }

            try:
                self.output_queue.put_nowait((redacted, pii_matches, stats))
            except queue.Full:
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.output_queue.put_nowait((redacted, pii_matches, stats))
                except queue.Full:
                    pass

            # FPS tracking
            now = time.perf_counter()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            self._pipeline_fps = len(frame_times)

    # ==================================================================
    # Custom regex callback
    # ==================================================================

    def add_custom_regex(self, label: str, pattern: str):
        """Add a custom regex pattern to the PII detector."""
        self.pii_detector.add_custom_pattern(label, pattern)
        print(f"[Main] Added custom pattern: {label} = {pattern}")

    # ==================================================================
    # Recording callback
    # ==================================================================

    def toggle_recording(self, should_record: bool):
        """Toggle video recording of the output stream."""
        if should_record:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"visionsecure_recording_{timestamp}.mp4"
            self.output_window.start_recording(filepath)
            print(f"[Main] Recording started: {filepath}")
        else:
            self.output_window._stop_recording()
            print("[Main] Recording stopped.")

    # ==================================================================
    # Stats polling for GUI
    # ==================================================================

    def get_stats(self):
        """Return current pipeline stats for the control panel."""
        return {
            "fps": self._pipeline_fps,
            "frame_redactions": self.redactor.frame_redaction_count,
            "session_total": self.redactor.session_redaction_count,
        }


def main():
    """Launch VisionSecure: control panel + output window."""
    print("=" * 60)
    print("  VisionSecure - Real-Time PII Redaction Agent v1.0")
    print("  Press 'Start' in the Control Panel to begin.")
    print("  Press 'Q' in the Output Window to quit.")
    print("  Press 'S' in the Output Window to take a screenshot.")
    print("")
    print("  TIP: The app captures the LEFT HALF of your screen by")
    print("  default. Place the content you want to monitor there.")
    print("  The Output window and Control Panel stay on the right.")
    print("=" * 60)

    app = VisionSecureApp()

    # ------------------------------------------------------------------
    # Tkinter GUI (must run on main thread)
    # ------------------------------------------------------------------
    root = tk.Tk()

    panel = ControlPanel(
        root=root,
        config=app.config,
        on_start=app.start_pipeline,
        on_pause=app.pause_pipeline,
        on_stop=app.stop_pipeline,
        on_add_custom_regex=app.add_custom_regex,
        on_record_toggle=app.toggle_recording,
    )

    # Periodic stats update
    def update_stats():
        try:
            stats = app.get_stats()
            panel.update_stats(
                fps=stats["fps"],
                frame_redactions=stats["frame_redactions"],
                session_total=stats["session_total"],
            )
        except Exception:
            pass
        root.after(500, update_stats)

    root.after(1000, update_stats)

    # Graceful shutdown
    def on_closing():
        app.stop_pipeline()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start tkinter event loop
    root.mainloop()

    # Ensure everything is cleaned up
    app.stop_pipeline()
    print("[Main] VisionSecure shutdown complete.")


if __name__ == "__main__":
    main()
