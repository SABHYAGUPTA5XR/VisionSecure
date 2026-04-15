"""
VisionSecure — STAGE 1: Frame Acquisition
High-performance screen capture using mss.
Runs in a dedicated daemon thread, pushing frames to a bounded queue.
Implements adaptive frame skipping when processing can't keep up.
Supports custom capture region to avoid the "hall of mirrors" effect.
"""

import time
import threading
import queue
import numpy as np
import mss


class ScreenCapture:
    """
    Threaded screen capture engine.
    Grabs the screen at target FPS and pushes frames into a shared queue.
    Uses drop-oldest policy to maintain low latency.
    """

    def __init__(self, config: dict, frame_queue: queue.Queue):
        """
        Args:
            config: Shared configuration dict (thread-safe reads).
                    Supports 'capture_region' key: dict with top/left/width/height
                    to capture a specific screen area instead of the full monitor.
            frame_queue: Bounded queue for captured frames.
        """
        self.config = config
        self.frame_queue = frame_queue
        self._thread = None
        self._running = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # Not paused by default

        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.skip_count = 0

    # ------------------------------------------------------------------
    # STAGE 1: Frame Acquisition — Start / Stop / Pause
    # ------------------------------------------------------------------

    def start(self):
        """Launch the capture thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._paused.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the capture thread to stop."""
        self._running.clear()
        self._paused.set()  # Unblock if paused
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self):
        """Pause frame capture."""
        self._paused.clear()

    def resume(self):
        """Resume frame capture."""
        self._paused.set()

    @property
    def is_running(self):
        return self._running.is_set()

    @property
    def is_paused(self):
        return not self._paused.is_set()

    # ------------------------------------------------------------------
    # STAGE 1: Frame Acquisition — Core capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """
        Main capture loop. Grabs screen at target FPS.
        Implements adaptive frame skipping if latency > 40ms.
        Supports capturing a custom region to avoid the recursive
        "hall of mirrors" effect when the output window is on the
        same monitor as the capture area.
        """
        sct = mss.mss()
        last_time = time.perf_counter()
        frame_times = []
        skip_next = False

        while self._running.is_set():
            # Honour pause
            self._paused.wait()
            if not self._running.is_set():
                break

            target_fps = self.config.get("target_fps", 20)
            target_interval = 1.0 / max(target_fps, 1)

            # STAGE 1: Adaptive frame skipping
            if skip_next:
                skip_next = False
                self.skip_count += 1
                time.sleep(target_interval * 0.5)
                continue

            capture_start = time.perf_counter()

            # ----------------------------------------------------------
            # STAGE 1: Determine capture region.
            # If 'capture_region' is set in config, use that custom region.
            # Otherwise, capture the full selected monitor.
            # Custom regions avoid capturing the VisionSecure output window
            # (preventing the "hall of mirrors" recursive capture).
            # ----------------------------------------------------------
            capture_region = self.config.get("capture_region", None)

            if capture_region and all(
                k in capture_region for k in ("top", "left", "width", "height")
            ):
                monitor = {
                    "top": capture_region["top"],
                    "left": capture_region["left"],
                    "width": capture_region["width"],
                    "height": capture_region["height"],
                }
            else:
                # Full monitor capture
                monitor_index = self.config.get("monitor_index", 1)
                monitors = sct.monitors
                if monitor_index < len(monitors):
                    monitor = monitors[monitor_index]
                else:
                    monitor = monitors[1] if len(monitors) > 1 else monitors[0]

            # STAGE 1: Capture the screen at highest achievable FPS
            try:
                raw = sct.grab(monitor)
                frame = np.array(raw, dtype=np.uint8)  # BGRA format
            except Exception:
                time.sleep(0.05)
                continue

            capture_elapsed = time.perf_counter() - capture_start

            # Push to queue using drop-oldest policy for low latency
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # Drop oldest
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            self.frame_count += 1

            # FPS calculation (rolling window)
            now = time.perf_counter()
            frame_times.append(now)
            # Keep only the last 30 timestamps
            frame_times = [t for t in frame_times if now - t < 1.0]
            self.fps = len(frame_times)

            # STAGE 1: Adaptive frame skipping — skip if latency > 40ms
            if capture_elapsed > 0.040:
                skip_next = True

            # Throttle to target FPS
            elapsed = now - last_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.perf_counter()

    @staticmethod
    def get_available_monitors():
        """Return list of available monitor specs."""
        with mss.mss() as sct:
            return sct.monitors
