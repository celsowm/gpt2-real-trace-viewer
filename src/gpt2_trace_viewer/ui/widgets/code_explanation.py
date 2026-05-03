from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gpt2_trace_viewer.domain.trace_step import TraceStep


class ExplanationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet("background-color: #1A1A1A;")
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QLabel("Explicação")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #666666; text-transform: uppercase; letter-spacing: 2px;")
        layout.addWidget(header)

        self._name_label = QLabel("")
        self._name_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self._name_label.setStyleSheet("color: #00FFFF;")
        self._name_label.setWordWrap(True)
        layout.addWidget(self._name_label)

        self._kind_label = QLabel("")
        self._kind_label.setFont(QFont("Arial", 10))
        self._kind_label.setStyleSheet("color: #AAAAAA;")
        layout.addWidget(self._kind_label)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #333333;")
        layout.addWidget(sep)

        self._desc_label = QLabel("")
        self._desc_label.setFont(QFont("Arial", 10))
        self._desc_label.setStyleSheet("color: #D4D4D4;")
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label)

        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background-color: #333333;")
        layout.addWidget(sep2)

        stats_header = QLabel("Tensor Stats")
        stats_header.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        stats_header.setStyleSheet("color: #666666;")
        layout.addWidget(stats_header)

        self._stats_label = QLabel("")
        self._stats_label.setFont(QFont("Consolas", 10))
        self._stats_label.setStyleSheet("color: #00FF00;")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        layout.addStretch()

        self._clear_view()

    def show_step(self, step: TraceStep, line: int) -> None:
        self._name_label.setText(f"Linha {line + 1}: {step.name}")
        self._kind_label.setText(step.kind)

        desc = step.description or "—"
        self._desc_label.setText(desc)

        stats_lines = []
        for k, v in step.stats.items():
            stats_lines.append(f"▸ {k}:  {v}")
        self._stats_label.setText("\n".join(stats_lines))

    def clear(self) -> None:
        self._clear_view()

    def _clear_view(self) -> None:
        self._name_label.setText("")
        self._kind_label.setText("")
        self._desc_label.setText("Pressione ▶ Reproduzir para ver a explicação de cada passo do forward real.")
        self._stats_label.setText("")
