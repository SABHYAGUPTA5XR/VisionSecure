"""
VisionSecure — STAGE 6: Frame Output & HUD Overlay
Displays the redacted frame in a dedicated OpenCV window.
Runs in its own thread to keep the GUI responsive.
"""

import time
import threading
import queue
import cv2
import numpy as np
from typing import List, Optional
from pii_detector import PIIMatch, PII_COLORS


class OutputWindow:
    """
    STAGE 6: Dedicated output window for the redacted screen stream.
    Overlays an HUD with live stats and colour-coded PII bounding boxes.
    """

    WINDOW_NAME = "VisionSecure — Redacted Output"

    def __init__(self, config: dict, output_queue: queue.Queue):
        """
        Args:
            config: Shared configuration dict.
            output_queue: Queue receiving (frame, pii_matches, stats) tuples.
        """
        self.config = config
        self.output_queue = output_queue
        self._thread = None
        self._running = threading.Event()

        # Recording
        self._video_writer = None
        self._recording = False

        # Stats
        self.display_fps = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Launch the output window thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the output window."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop_recording()

    # ------------------------------------------------------------------
    # Recording controls
    # ------------------------------------------------------------------

    def start_recording(self, filepath: str, fps: float = 20.0, frame_size: tuple = None):
        """Start recording the redacted output to a video file."""
        if self._recording:
            return
        if frame_size is None:
            frame_size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
        self._recording = True

    def _stop_recording(self):
        """Stop and release the video writer."""
        self._recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    # ------------------------------------------------------------------
    # STAGE 6: Main display loop
    # ------------------------------------------------------------------

    def _display_loop(self):
        """Main loop: pull redacted frames from queue, overlay HUD, display."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 800, 500)
        # Position at right side of screen to avoid being captured
        # by the default "left half" capture region.
        cv2.moveWindow(self.WINDOW_NAME, 970, 50)

        frame_times = []
        last_frame = None

        while self._running.is_set():
            try:
                data = self.output_queue.get(timeout=0.1)
            except queue.Empty:
                # Show last frame if no new data
                if last_frame is not None:
                    cv2.imshow(self.WINDOW_NAME, last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running.clear()
                    break
                continue

            frame, pii_matches, stats = data

            # ------------------------------------------------------------------
            # STAGE 6: Draw colour-coded bounding boxes with PII labels
            # ------------------------------------------------------------------
            show_boxes = self.config.get("show_bounding_boxes", True)
            if show_boxes and pii_matches:
                frame = self._draw_annotations(frame, pii_matches)

            # ------------------------------------------------------------------
            # STAGE 6: Overlay live HUD
            # ------------------------------------------------------------------
            frame = self._draw_hud(frame, stats)

            # FPS tracking
            now = time.perf_counter()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            self.display_fps = len(frame_times)

            # Display
            cv2.imshow(self.WINDOW_NAME, frame)
            last_frame = frame

            # Recording
            if self._recording and self._video_writer is not None:
                self._video_writer.write(frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._running.clear()
                break
            elif key == ord("s"):
                # Screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"visionsecure_screenshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"[Output] Screenshot saved: {filename}")

        cv2.destroyWindow(self.WINDOW_NAME)

    # ------------------------------------------------------------------
    # STAGE 6: HUD overlay rendering
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """
        STAGE 6: Overlay HUD showing FPS, frame redactions, session total.
        Draws a semi-transparent dark background with white text in the corner.
        """
        fps = stats.get("fps", 0)
        frame_redactions = stats.get("frame_redactions", 0)
        session_total = stats.get("session_total", 0)

        hud_texts = [
            f"FPS: {fps:.0f}",
            f"Redactions: {frame_redactions}",
            f"Session: {session_total}",
        ]

        # Semi-transparent background
        padding = 10
        line_height = 28
        hud_w = 240
        hud_h = padding * 2 + line_height * len(hud_texts)

        overlay = frame.copy()
        cv2.rectangle(
            overlay, (10, 10), (10 + hud_w, 10 + hud_h),
            (0, 0, 0), -1
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw HUD text
        for i, text in enumerate(hud_texts):
            y = 10 + padding + (i + 1) * line_height - 5
            cv2.putText(
                frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2,
                cv2.LINE_AA
            )

        return frame

    # ------------------------------------------------------------------
    # STAGE 6: Colour-coded PII annotations
    # ------------------------------------------------------------------

    def _draw_annotations(
        self, frame: np.ndarray, pii_matches: List[PIIMatch]
    ) -> np.ndarray:
        """
        STAGE 6: Draw colour-coded bounding boxes and PII type labels.
        Colour coding:
            RED=email, ORANGE=phone, YELLOW=IP, PURPLE=card,
            BLUE=API key, GREEN=custom, CYAN=Aadhaar, PINK=PAN
        """
        for match in pii_matches:
            x, y, w, h = match.bbox
            color = match.color

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw PII type label above the box
            label = match.pii_type
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y - 8, label_size[1] + 4)

            # Background for label
            cv2.rectangle(
                frame,
                (x, label_y - label_size[1] - 4),
                (x + label_size[0] + 4, label_y + 4),
                color, -1
            )
            cv2.putText(
                frame, label, (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA
            )

        return frame
