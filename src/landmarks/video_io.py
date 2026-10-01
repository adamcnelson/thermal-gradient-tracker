"""
Frame-accurate RGB video access.

cv2's CAP_PROP_POS_FRAMES seek is NOT frame-accurate on every file in this corpus:
on 08-07-25_Test_7/2025-08-07_10-54-09.mp4 (clean CFR 60fps, PTS == idx/60) every
seek lands exactly 1.69% late (+1.8s at t=106s, +33s at t=1965s), verified by
pixel-exact matching against sequential decode (2026-10-01); Test_3/Test_4 seek
exactly. Sequential grab() is the only reliable way to reach a given frame index.
"""

import cv2


class SequentialFrameReader:
    """Reads frames by index via forward-only sequential decode.

    Requests should be made in non-decreasing index order; a request behind the
    current position reopens the video and decodes from the start (correct, slow).
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = None
        self._next_idx = 0
        self._open()
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _open(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")
        self._next_idx = 0

    def read(self, idx: int):
        """Return (ok, frame) for frame `idx` (0-based)."""
        if idx < self._next_idx:
            self._open()
        while self._next_idx < idx:
            if not self._cap.grab():
                return False, None
            self._next_idx += 1
        ok, frame = self._cap.read()
        if ok:
            self._next_idx += 1
        return ok, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
