from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class StepRevealController(QObject):
    step_revealed = pyqtSignal(int)
    finished = pyqtSignal()
    started = pyqtSignal(int)
    paused = pyqtSignal()
    resumed = pyqtSignal()

    def __init__(self, total_steps: int, delay_ms: int = 500) -> None:
        super().__init__()
        self._total = total_steps
        self._current = -1
        self._delay = delay_ms
        self._running = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    @property
    def current_index(self) -> int:
        return self._current

    @property
    def total_steps(self) -> int:
        return self._total

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_finished(self) -> bool:
        return self._current >= self._total - 1

    def set_delay(self, ms: int) -> None:
        self._delay = max(50, ms)
        if self._running:
            self._timer.setInterval(self._delay)

    def start(self) -> None:
        if self._total == 0:
            return
        self._running = True
        self._current = -1
        self.started.emit(self._total)
        self._timer.start(self._delay)

    def resume(self) -> None:
        if not self._running and not self.is_finished:
            self._running = True
            self._timer.start(self._delay)
            self.resumed.emit()

    def pause(self) -> None:
        if self._running:
            self._running = False
            self._timer.stop()
            self.paused.emit()

    def stop(self) -> None:
        self._running = False
        self._timer.stop()

    def jump_to(self, index: int) -> None:
        index = max(-1, min(index, self._total - 1))
        self._current = index
        self.step_revealed.emit(index)
        if index >= self._total - 1:
            self.stop()
            self.finished.emit()

    def jump_to_end(self) -> None:
        self.jump_to(self._total - 1)

    def _advance(self) -> None:
        next_idx = self._current + 1
        if next_idx >= self._total:
            self.stop()
            self.finished.emit()
            return
        self._current = next_idx
        self.step_revealed.emit(next_idx)
        if next_idx >= self._total - 1:
            self.stop()
            self.finished.emit()
