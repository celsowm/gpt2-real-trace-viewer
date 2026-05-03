from __future__ import annotations

import math

import torch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gpt2_trace_viewer.application.trace_result import TraceResult
from gpt2_trace_viewer.domain.tensor_inspector import safe_float
from gpt2_trace_viewer.domain.trace_step import TraceStep
from gpt2_trace_viewer.ui.step_reveal import StepRevealController


class RealNeuralGraphViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.result: TraceResult | None = None
        self._controller: StepRevealController | None = None
        self._revealed_nodes: list[dict] = []
        self._all_node_boxes: list[dict] = []
        self._highlight_item: QGraphicsRectItem | None = None
        self._highlight_timer: QTimer | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()

        self.play_btn = QPushButton("Reproduzir")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        self._style_button(self.play_btn)

        self.skip_end_btn = QPushButton("Ir para o fim")
        self.skip_end_btn.clicked.connect(self._jump_to_end)
        self.skip_end_btn.setEnabled(False)
        self._style_button(self.skip_end_btn)

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
        controls.addWidget(QLabel("Vel:"))
        controls.addWidget(self.speed_slider)
        controls.addWidget(self.step_label)
        controls.addStretch()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Bloco selecionado", "Fluxo completo", "Somente global"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(QLabel("Modo:"))
        controls.addWidget(self.mode_combo)

        self.block_combo = QComboBox()
        self.block_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(QLabel("Bloco:"))
        controls.addWidget(self.block_combo)

        zoom_in = QPushButton("Zoom +")
        zoom_in.clicked.connect(lambda: self.view.scale(1.15, 1.15))
        controls.addWidget(zoom_in)

        zoom_out = QPushButton("Zoom -")
        zoom_out.clicked.connect(lambda: self.view.scale(1 / 1.15, 1 / 1.15))
        controls.addWidget(zoom_out)

        fit = QPushButton("Ajustar")
        fit.clicked.connect(self.fit_to_view)
        controls.addWidget(fit)

        controls.addStretch()
        layout.addLayout(controls)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(QColor("#0F0F0F"))
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        layout.addWidget(self.view)

        info = QLabel(
            "Grafo real: nos e setas sao derivados das etapas do forward pass. "
            "A espessura da seta usa a norma L2 real do tensor de destino."
        )
        info.setStyleSheet("color: #00FFFF; padding: 6px;")
        layout.addWidget(info)

    def _style_button(self, btn: QPushButton) -> None:
        btn.setStyleSheet(
            "QPushButton { background-color: #007ACC; color: white; padding: 6px 16px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0098FF; }"
            "QPushButton:disabled { background-color: #555555; color: #AAAAAA; }"
        )

    def set_result(self, result: TraceResult) -> None:
        self.result = result
        self.block_combo.blockSignals(True)
        self.block_combo.clear()
        blocks = sorted({step.block for step in result.steps if step.block is not None})
        for block in blocks:
            self.block_combo.addItem(f"Bloco {block}", block)
        self.block_combo.blockSignals(False)

        n = len(self._filtered_steps())
        self._controller = StepRevealController(n, delay_ms=self._delay_from_slider())
        self._controller.step_revealed.connect(self._on_step_revealed)
        self._controller.finished.connect(self._on_reveal_finished)
        self._controller.started.connect(self._on_reveal_started)
        self._controller.paused.connect(self._on_reveal_paused)
        self._controller.resumed.connect(self._on_reveal_resumed)

        self.play_btn.setEnabled(True)
        self.skip_end_btn.setEnabled(True)
        self.step_label.setText(f"0 / {n} nos")
        self._update_play_button_text()
        self.render_graph()

    def reset(self) -> None:
        if self._controller:
            self._controller.stop()
        self._controller = None
        self.result = None
        self._revealed_nodes.clear()
        self._all_node_boxes.clear()
        self.block_combo.clear()
        self.scene.clear()
        self.play_btn.setEnabled(False)
        self.skip_end_btn.setEnabled(False)
        self.step_label.setText("")

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
            self.play_btn.setText("Reproduzir")
        elif self._controller.is_running:
            self.play_btn.setText("Pausar")
        elif self._controller.is_finished:
            self.play_btn.setText("Repetir")
        else:
            self.play_btn.setText("Retomar")

    def _jump_to_end(self) -> None:
        if self._controller:
            self._controller.jump_to_end()

    def _on_mode_changed(self) -> None:
        if self._controller:
            self._controller.stop()
        self._revealed_nodes.clear()
        self._all_node_boxes.clear()
        if self.result is not None:
            self.set_result(self.result)
        else:
            self.scene.clear()

    def _on_reveal_started(self, total: int) -> None:
        self._revealed_nodes.clear()
        self.step_label.setText(f"0 / {total} nos")
        self._update_play_button_text()
        self.scene.clear()
        self._draw_title()
        self._draw_instructions()
        self._advance_to_node(0)

    def _on_reveal_finished(self) -> None:
        self._clear_highlight()
        self._update_play_button_text()

    def _on_reveal_paused(self) -> None:
        self._update_play_button_text()

    def _on_reveal_resumed(self) -> None:
        self._update_play_button_text()

    def _on_step_revealed(self, index: int) -> None:
        if index < 0 or index >= len(self._all_node_boxes):
            return
        self._revealed_nodes.append(self._all_node_boxes[index])
        self._advance_to_node(len(self._revealed_nodes))
        self.step_label.setText(f"{len(self._revealed_nodes)} / {len(self._all_node_boxes)} nos")

    def _advance_to_node(self, count: int) -> None:
        self.scene.clear()
        self._draw_title()

        revealed = self._all_node_boxes[:count]
        if not revealed:
            self._draw_instructions()
            return

        max_l2 = max(
            [safe_float(nd["step"].stats.get("l2"), 0.0) for nd in revealed] or [1.0]
        )

        for nd in revealed:
            self._draw_single_node(nd)

        for i in range(1, len(revealed)):
            self._draw_edge(revealed[i - 1], revealed[i], max_l2)

        last = revealed[-1]
        self._highlight_node(last)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-80, -80, 120, 120))
        self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80),
                            Qt.AspectRatioMode.KeepAspectRatio)

    def _highlight_node(self, nd: dict) -> None:
        self._highlight_item = QGraphicsRectItem(
            nd["x"] - 3, nd["y"] - 3, nd["w"] + 6, nd["h"] + 6
        )
        self._highlight_item.setPen(QPen(QColor("#00FFFF"), 3))
        self._highlight_item.setZValue(10)
        self.scene.addItem(self._highlight_item)

        if self._highlight_timer:
            self._highlight_timer.stop()
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._clear_highlight)
        self._highlight_timer.start(300)

    def _clear_highlight(self) -> None:
        if self._highlight_item:
            try:
                self.scene.removeItem(self._highlight_item)
            except RuntimeError:
                pass
            self._highlight_item = None

    def _draw_title(self) -> None:
        title = self.scene.addText("Grafo Neural Real do GPT-2")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#FFD700"))
        title.setPos(40, 20)

        subtitle = self.scene.addText(
            "Cada no e uma operacao real capturada no forward."
        )
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setDefaultTextColor(QColor("#AAAAAA"))
        subtitle.setPos(40, 52)

    def _draw_instructions(self) -> None:
        text = self.scene.addText("Pressione Reproduzir para comecar a revelacao gradual.")
        text.setFont(QFont("Arial", 14))
        text.setDefaultTextColor(QColor("#FFFFFF"))
        text.setPos(40, 100)

    def _draw_single_node(self, nd: dict) -> None:
        x, y, width, height = nd["x"], nd["y"], nd["w"], nd["h"]
        step: TraceStep = nd["step"]
        color = self._node_color(step.kind)

        rect = QGraphicsRectItem(x, y, width, height)
        rect.setPen(QPen(QColor("#FFFFFF"), 1.1))
        rect.setBrush(QBrush(color))
        rect.setZValue(2)
        self.scene.addItem(rect)

        title = self.scene.addText(f"{nd['index']}. {step.kind}")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor("#FFFFFF"))
        title.setPos(x + 10, y + 5)
        title.setZValue(3)

        name = step.name if len(step.name) <= 38 else step.name[:35] + "..."
        name_text = self.scene.addText(name)
        name_text.setFont(QFont("Consolas", 7))
        name_text.setDefaultTextColor(QColor("#E0E0E0"))
        name_text.setPos(x + 10, y + 22)
        name_text.setZValue(3)

        stats_text = (
            f"shape: {step.stats['shape']}\n"
            f"mean: {step.stats['mean']} | std: {step.stats['std']}\n"
            f"L2: {step.stats['l2']}"
        )
        stats = self.scene.addText(stats_text)
        stats.setFont(QFont("Consolas", 7))
        stats.setDefaultTextColor(QColor("#FFFFFF"))
        stats.setPos(x + 10, y + 40)
        stats.setZValue(3)

        self._draw_mini_heatmap(x, y, step)
        self._draw_neuron_bar(x, y, step)

        rect.setToolTip(
            f"{step.name}\n"
            f"Tipo: {step.kind}\n"
            f"Bloco: {step.block if step.block is not None else 'Global'}\n"
            f"Shape: {step.stats['shape']}\n"
            f"Min: {step.stats['min']}\n"
            f"Max: {step.stats['max']}\n"
            f"Mean: {step.stats['mean']}\n"
            f"Std: {step.stats['std']}\n"
            f"L2: {step.stats['l2']}\n\n"
            f"{step.description}"
        )

    def _sample_tensor_2d(self, tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor | None:
        t = tensor.detach().float().cpu()
        flat = t.reshape(-1)
        if flat.numel() == 0:
            return None
        if flat.numel() < rows * cols:
            sampled = torch.zeros(rows * cols)
            sampled[:flat.numel()] = flat
            sampled[flat.numel():] = flat.mean()
            return sampled.view(rows, cols)
        indices = torch.linspace(0, flat.numel() - 1, rows * cols, dtype=torch.long)
        return flat[indices].view(rows, cols)

    def _draw_mini_heatmap(self, x: float, y: float, step: TraceStep) -> None:
        grid = self._sample_tensor_2d(step.tensor, 4, 8)
        if grid is None:
            return
        vmin = grid.min()
        vrange = grid.max() - vmin
        if vrange == 0:
            vrange = 1
        cell_w = 5
        cell_h = 5
        ox = x + 190
        oy = y + 74
        for r in range(4):
            for c in range(8):
                val = grid[r, c].item()
                norm = (val - vmin) / vrange
                r8 = int(min(255, norm * 255))
                g8 = int(min(255, (1 - abs(norm - 0.5) * 2) * 255))
                b8 = int(min(255, (1 - norm) * 255))
                cell = QGraphicsRectItem(ox + c * (cell_w + 1), oy + r * (cell_h + 1), cell_w, cell_h)
                cell.setBrush(QBrush(QColor(r8, g8, b8)))
                cell.setPen(QPen(Qt.PenStyle.NoPen))
                cell.setZValue(4)
                self.scene.addItem(cell)

    def _draw_neuron_bar(self, x: float, y: float, step: TraceStep) -> None:
        t = step.tensor.detach().float().cpu()
        flat = t.reshape(-1)
        if flat.numel() == 0:
            return
        n = min(32, flat.numel())
        indices = torch.linspace(0, flat.numel() - 1, n, dtype=torch.long)
        vals = flat[indices]
        vmin = vals.min()
        vrange = vals.max() - vmin
        if vrange == 0:
            vrange = 1
        bar_w = 4
        bar_h = 24
        ox = x + 10
        oy = y + 68
        for i in range(n):
            norm = (vals[i].item() - vmin) / vrange
            r8 = int(min(255, norm * 255))
            g8 = int(min(255, (1 - abs(norm - 0.5) * 2) * 255))
            b8 = int(min(255, (1 - norm) * 255))
            h = max(2, int(norm * bar_h))
            bar = QGraphicsRectItem(ox + i * (bar_w + 1), oy + (bar_h - h), bar_w, h)
            bar.setBrush(QBrush(QColor(r8, g8, b8)))
            bar.setPen(QPen(Qt.PenStyle.NoPen))
            bar.setZValue(4)
            self.scene.addItem(bar)

    def fit_to_view(self) -> None:
        rect = self.scene.itemsBoundingRect().adjusted(-80, -80, 80, 80)
        if rect.isValid():
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _filtered_steps(self) -> list[TraceStep]:
        if self.result is None:
            return []
        mode = self.mode_combo.currentText()
        steps = self.result.steps
        if mode == "Somente global":
            return [step for step in steps if step.block is None]
        if mode == "Bloco selecionado":
            selected_block = self.block_combo.currentData()
            if selected_block is None:
                return []
            return [step for step in steps if step.block == selected_block]
        return steps

    def _node_color(self, kind: str) -> QColor:
        lowered = kind.lower()
        if "embedding" in lowered:
            return QColor("#007ACC")
        if "layernorm" in lowered:
            return QColor("#7B1FA2")
        if any(term in lowered for term in ["attention", " q", " k", " v", "heads", "qk"]):
            return QColor("#00897B")
        if "softmax" in lowered:
            return QColor("#F57C00")
        if "residual" in lowered:
            return QColor("#546E7A")
        if "mlp" in lowered or "gelu" in lowered:
            return QColor("#C2185B")
        if "vocabulary" in lowered or "token" in lowered:
            return QColor("#388E3C")
        return QColor("#37474F")

    def render_graph(self) -> None:
        self._revealed_nodes.clear()
        self.scene.clear()
        self._draw_title()

        steps = self._filtered_steps()
        if not steps:
            text = self.scene.addText("Nenhuma etapa para exibir com o filtro atual.")
            text.setDefaultTextColor(QColor("#FFFFFF"))
            text.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            text.setPos(40, 100)
            return

        self._all_node_boxes = []
        max_l2 = max([safe_float(step.stats.get("l2"), 0.0) for step in steps] or [1.0])

        x_start = 50
        y_start = 115
        x_gap = 325
        y_gap = 145
        max_cols = 4

        current_group = object()
        group_extra_offset = 0
        count_in_group = 0

        for index, step in enumerate(steps):
            group = "Global" if step.block is None else f"Bloco {step.block}"
            if group != current_group:
                if index != 0:
                    rows_used = math.ceil(max(count_in_group, 1) / max_cols)
                    group_extra_offset += (rows_used + 1) * y_gap
                    count_in_group = 0
                current_group = group

            col = count_in_group % max_cols
            row = count_in_group // max_cols
            x = x_start + col * x_gap
            y = y_start + group_extra_offset + row * y_gap

            nd = {
                "x": x,
                "y": y,
                "w": 270,
                "h": 105,
                "step": step,
                "index": index,
                "max_l2": max_l2,
            }
            self._all_node_boxes.append(nd)
            count_in_group += 1

        self._draw_instructions()
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-80, -80, 120, 120))

    def _draw_edge(self, left_node: dict, right_node: dict, max_strength: float) -> None:
        x1 = float(left_node["x"]) + float(left_node["w"])
        y1 = float(left_node["y"]) + float(left_node["h"]) / 2
        x2 = float(right_node["x"])
        y2 = float(right_node["y"]) + float(right_node["h"]) / 2

        s = safe_float(right_node["step"].stats.get("l2"), 0.0)
        ratio = 0.0 if max_strength <= 0 else min(1.0, s / max_strength)
        width = 1.0 + ratio * 5.0
        alpha = 80 + int(ratio * 175)
        pen = QPen(QColor(0, 255, 180, alpha), width)

        line = self.scene.addLine(x1, y1, x2, y2, pen)
        line.setZValue(1)

        arrow_size = 8 + ratio * 7
        angle = math.atan2(y2 - y1, x2 - x1)

        p1x = x2 - arrow_size * math.cos(angle - math.pi / 6)
        p1y = y2 - arrow_size * math.sin(angle - math.pi / 6)
        p2x = x2 - arrow_size * math.cos(angle + math.pi / 6)
        p2y = y2 - arrow_size * math.sin(angle + math.pi / 6)

        self.scene.addLine(x2, y2, p1x, p1y, pen).setZValue(1)
        self.scene.addLine(x2, y2, p2x, p2y, pen).setZValue(1)
