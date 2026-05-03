from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gpt2_trace_viewer.application.trace_result import AttentionRecord, TraceResult
from gpt2_trace_viewer.ui.widgets.attention_matrix_viewer import AttentionMatrixViewer


class AttentionTab(QWidget):
    """Controls and canvas for real attention matrices."""

    def __init__(self) -> None:
        super().__init__()
        self.result: TraceResult | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Bloco:"))
        self.block_combo = QComboBox()
        self.block_combo.currentIndexChanged.connect(self._draw_selected_attention)
        controls.addWidget(self.block_combo)

        controls.addWidget(QLabel("Head:"))
        self.head_combo = QComboBox()
        self.head_combo.currentIndexChanged.connect(self._draw_selected_attention)
        controls.addWidget(self.head_combo)

        controls.addStretch()
        layout.addLayout(controls)

        self.viewer = AttentionMatrixViewer()
        layout.addWidget(self.viewer)

    def set_result(self, result: TraceResult) -> None:
        self.result = result
        self._populate_controls(result.attention_records)
        self._draw_selected_attention()

    def reset(self) -> None:
        self.result = None
        self.block_combo.clear()
        self.head_combo.clear()
        self.viewer.clear()

    def _populate_controls(self, records: list[AttentionRecord]) -> None:
        blocks = sorted({record.block for record in records})
        heads = sorted({record.head for record in records})

        self.block_combo.blockSignals(True)
        self.head_combo.blockSignals(True)
        self.block_combo.clear()
        self.head_combo.clear()

        for block in blocks:
            self.block_combo.addItem(str(block), block)

        for head in heads:
            self.head_combo.addItem(str(head), head)

        self.block_combo.blockSignals(False)
        self.head_combo.blockSignals(False)

    def _draw_selected_attention(self) -> None:
        if self.result is None:
            return

        block = self.block_combo.currentData()
        head = self.head_combo.currentData()

        if block is None or head is None:
            return

        for record in self.result.attention_records:
            if record.block == block and record.head == head:
                self.viewer.draw_attention(
                    tokens=self.result.tokens,
                    matrix=record.matrix.tolist(),
                )
                return
