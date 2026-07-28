"""Terminal logging and progress bar utilities."""

import sys
import time


def setup_logger(name: str = "thermal-tracker"):
    import logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


class ProgressBar:
    """Minimal terminal progress bar — no extra dependencies."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self._start = time.time()
        self._last_print = 0.0

    def update(self, current: int, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_print) < 1.0:
            return
        self._last_print = now
        frac = current / max(self.total, 1)
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = now - self._start
        eta = (elapsed / frac - elapsed) if frac > 0 else 0
        sys.stdout.write(
            f"\r{self.desc}  [{bar}] {current}/{self.total}  "
            f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s"
        )
        sys.stdout.flush()

    def done(self) -> None:
        elapsed = time.time() - self._start
        bar = "█" * self.width
        sys.stdout.write(
            f"\r{self.desc}  [{bar}] {self.total}/{self.total}  "
            f"done in {elapsed:.1f}s          \n"
        )
        sys.stdout.flush()
