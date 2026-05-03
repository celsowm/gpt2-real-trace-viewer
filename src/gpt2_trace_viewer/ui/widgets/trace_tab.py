from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gpt2_trace_viewer.application.trace_result import TraceResult
from gpt2_trace_viewer.domain.tensor_inspector import top_abs_values
from gpt2_trace_viewer.ui.step_reveal import StepRevealController


class TraceTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.result: TraceResult | None = None
        self._controller: StepRevealController | None = None
        self._tree_items: list[QTreeWidgetItem] = []

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶ Reproduzir")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(
            "QPushButton { background-color: #007ACC; color: white; padding: 6px 16px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0098FF; }"
            "QPushButton:disabled { background-color: #555555; color: #AAAAAA; }"
        )

        self.skip_end_btn = QPushButton("⏭ Ir para o fim")
        self.skip_end_btn.clicked.connect(self._jump_to_end)
        self.skip_end_btn.setEnabled(False)
        self.skip_end_btn.setStyleSheet(self.play_btn.styleSheet())

        self.step_label = QLabel("")
        self.step_label.setFont(QFont("Consolas", 10))
        self.step_label.setStyleSheet("color: #00FFFF;")

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self._on_speed_change)

        controls.addWidget(self.play_btn)
        controls.addWidget(self.skip_end_btn)
        controls.addWidget(QLabel("Velocidade:"))
        controls.addWidget(self.speed_slider)
        controls.addWidget(self.step_label)
        controls.addStretch()
        layout.addLayout(controls)

        inner = QHBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Etapa", "Tipo", "Bloco", "Shape", "Mean", "Std", "L2"])
        self.tree.itemSelectionChanged.connect(self._on_step_selected)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                background-color: #111111;
                color: #DDDDDD;
                font-family: Consolas;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #333333;
                color: white;
            }
            """
        )
        inner.addWidget(self.tree, 3)

        right_layout = QVBoxLayout()
        self.detail = QTextBrowser()
        self.detail.setStyleSheet(
            "background-color: #0B0B0B; color: #DDDDDD; font-family: Consolas;"
        )
        right_layout.addWidget(self.detail, 2)

        self.top_table = QTableWidget()
        self.top_table.setColumnCount(3)
        self.top_table.setHorizontalHeaderLabels(["Índice", "Valor", "|Valor|"])
        self.top_table.setStyleSheet(
            "background-color: #111111; color: #DDDDDD; font-family: Consolas;"
        )
        right_layout.addWidget(self.top_table, 1)

        inner.addLayout(right_layout, 2)
        layout.addLayout(inner, 1)

    def reset(self) -> None:
        if self._controller:
            self._controller.stop()
        self._controller = None
        self.result = None
        self._tree_items.clear()
        self.tree.clear()
        self.detail.clear()
        self.top_table.setRowCount(0)
        self.play_btn.setEnabled(False)
        self.skip_end_btn.setEnabled(False)
        self.step_label.setText("")

    def set_result(self, result: TraceResult) -> None:
        self.result = result
        self._populate_tree()
        n = len(result.steps)
        self._controller = StepRevealController(n, delay_ms=self._delay_from_slider())
        self._controller.step_revealed.connect(self._on_step_revealed)
        self._controller.finished.connect(self._on_reveal_finished)
        self._controller.started.connect(self._on_reveal_started)
        self._controller.paused.connect(self._on_reveal_paused)
        self._controller.resumed.connect(self._on_reveal_resumed)
        self.play_btn.setEnabled(True)
        self.skip_end_btn.setEnabled(True)
        self.step_label.setText(f"0 / {n} etapas")
        self._update_play_button_text()

    def _delay_from_slider(self) -> int:
        val = self.speed_slider.value()
        return int(1000 / val)

    def _on_speed_change(self) -> None:
        if self._controller:
            self._controller.set_delay(self._delay_from_slider())

    def _toggle_play(self) -> None:
        if self._controller is None:
            return
        if self._controller.is_running:
            self._controller.pause()
        elif self._controller.is_finished:
            self._controller.start()
        else:
            self._controller.resume()

    def _update_play_button_text(self) -> None:
        if self._controller is None:
            self.play_btn.setText("▶ Reproduzir")
        elif self._controller.is_running:
            self.play_btn.setText("⏸ Pausar")
        elif self._controller.is_finished:
            self.play_btn.setText("↻ Repetir")
        else:
            self.play_btn.setText("▶ Retomar")

    def _jump_to_end(self) -> None:
        if self._controller:
            self._controller.jump_to_end()

    def _on_reveal_started(self, total: int) -> None:
        self.step_label.setText(f"0 / {total} etapas")
        self._update_play_button_text()

    def _on_reveal_finished(self) -> None:
        self._update_play_button_text()

    def _on_reveal_paused(self) -> None:
        self._update_play_button_text()

    def _on_reveal_resumed(self) -> None:
        self._update_play_button_text()

    def _on_step_revealed(self, index: int) -> None:
        if self.result is None or index < 0 or index >= len(self._tree_items):
            return
        item = self._tree_items[index]
        self.tree.scrollToItem(item)
        self.tree.setCurrentItem(item)
        item.setSelected(True)
        self.step_label.setText(f"{index + 1} / {len(self._tree_items)} etapas")

    def _populate_tree(self) -> None:
        self.tree.clear()
        self._tree_items.clear()

        if self.result is None:
            return

        roots: dict[str, QTreeWidgetItem] = {}

        for index, step in enumerate(self.result.steps):
            root_name = "Global" if step.block is None else f"Bloco {step.block}"
            if root_name not in roots:
                root = QTreeWidgetItem([root_name, "", "", "", "", "", ""])
                self.tree.addTopLevelItem(root)
                roots[root_name] = root

            block_text = "" if step.block is None else str(step.block)
            item = QTreeWidgetItem(
                [
                    f"{index}. {step.name}" if index < 999 else step.name,
                    step.kind,
                    block_text,
                    step.stats["shape"],
                    step.stats["mean"],
                    step.stats["std"],
                    step.stats["l2"],
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            roots[root_name].addChild(item)
            self._tree_items.append(item)

        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

    def _on_step_selected(self) -> None:
        if self.result is None:
            return
        selected = self.tree.selectedItems()
        if not selected:
            return
        item = selected[0]
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is None:
            return
        step = self.result.steps[int(index)]

        self.detail.setHtml(
            f"""
            <h2 style="color:#FFD700;">{step.name}</h2>
            <p><b>Tipo:</b> {step.kind}</p>
            <p><b>Bloco:</b> {step.block if step.block is not None else "Global"}</p>
            <p><b>Descrição:</b> {step.description}</p>
            <h3 style="color:#00FFFF;">Estatísticas reais do tensor</h3>
            <table border="1" cellspacing="0" cellpadding="4">
                <tr><td>shape</td><td>{step.stats["shape"]}</td></tr>
                <tr><td>min</td><td>{step.stats["min"]}</td></tr>
                <tr><td>max</td><td>{step.stats["max"]}</td></tr>
                <tr><td>mean</td><td>{step.stats["mean"]}</td></tr>
                <tr><td>std</td><td>{step.stats["std"]}</td></tr>
                <tr><td>L2</td><td>{step.stats["l2"]}</td></tr>
                <tr><td>abs mean</td><td>{step.stats["abs_mean"]}</td></tr>
            </table>
            """
        )

        rows = top_abs_values(step.tensor, k=20)
        self.top_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self.top_table.setItem(row_index, 0, QTableWidgetItem(row["index"]))
            self.top_table.setItem(row_index, 1, QTableWidgetItem(row["value"]))
            self.top_table.setItem(row_index, 2, QTableWidgetItem(row["abs"]))
        self.top_table.resizeColumnsToContents()
