"""Threaded camera capture that always exposes the latest frame."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, source: int | str):
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        if isinstance(self.source, str):
            self.source = str(Path(self.source))

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("ERROR: Could not open camera source")
            sys.exit(1)

        print("Camera opened")

        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                    self.frame_id += 1
                continue

            if not ret:
                # For file sources loop at EOF; webcams will just continue next read.
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_with_id(self):
        with self.lock:
            return self.frame, self.frame_id

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()
