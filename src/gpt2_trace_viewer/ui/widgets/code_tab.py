from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gpt2_trace_viewer.application.trace_result import TraceResult
from gpt2_trace_viewer.ui.step_reveal import StepRevealController
from gpt2_trace_viewer.ui.widgets.code_explanation import ExplanationPanel
from gpt2_trace_viewer.ui.widgets.code_source_strings import CODE_SOURCE
from gpt2_trace_viewer.ui.widgets.code_syntax import PythonSyntaxHighlighter


LINE_TOOLTIPS: dict[int, str] = {
    1: "Residual guardado (atenção)",
    2: "LayerNorm antes da atenção (ln_1)",
    3: "Chama GPT2Attention.forward",
    15: "Residual guardado (MLP)",
    64: "c_attn = Linear(embed_dim, embed_dim*3) fundido QKV",
    71: "Q @ K.T com fator de escala 1/sqrt(head_dim)",
    73: "Soma a máscara causal (-inf nas posições proibidas)",
    74: "Softmax: normaliza para probabilidades de atenção",
    102: "Expansão MLP: embed_dim -> intermediate_size (4x)",
    154: "Token embedding via wte",
    156: "Position embedding via wpe",
}

CODE_LINES = CODE_SOURCE.split("\n")

STEP_TO_PATTERN: dict[str, str] = {
    "Token Embedding": "self.wte(input_ids)",
    "Position Embedding": "self.wpe(position_ids)",
    "Embedding Sum": "hidden_states = token_embeds + position_embeds",
    "Residual Input": "residual = hidden_states",
    "residual_attention_input": "residual = hidden_states",
    "residual_mlp_input": "residual = hidden_states",
    "LayerNorm": "self.ln_1",
    "ln_1": "self.ln_1(hidden_states)",
    "ln_2": "self.ln_2(hidden_states)",
    "Linear QKV fused": "self.c_attn = torch.nn.Linear",
    "c_attn": "qkv = self.c_attn(hidden_states)",
    "Q split": "query, key, value = qkv.split",
    "K split": "query, key, value = qkv.split",
    "V split": "query, key, value = qkv.split",
    "Q heads": "query = query.view(",
    "K heads": "key = key.view(",
    "V heads": "value = value.view(",
    "q_heads": "query = query.view(",
    "k_heads": "key = key.view(",
    "v_heads": "value = value.view(",
    "QK scores": "query @ key.transpose",
    "Causal Mask": "attn_weights = attn_weights + attention_mask",
    "Attention Softmax": "F.softmax(attn_weights, dim=-1)",
    "Attn Head": "attn_output = attn_weights @ value",
    "Merge Heads": "attn_output = attn_output.transpose(1, 2).contiguous()",
    "Attention Output Projection": "self.c_proj(attn_output)",
    "Residual Add": "hidden_states = residual + attn_output",
    "MLP Expand": "self.c_fc = torch.nn.Linear",
    "GELU": "self.act(hidden_states)",
    "MLP Project": "self.c_proj(hidden_states)",
    "Final LayerNorm": "self.ln_f(hidden_states)",
    "Last Token State": "self.transformer(input_ids)",
    "Vocabulary Projection": "self.lm_head(hidden_states)",
    "Vocabulary Softmax": "return logits",
}


def _build_pattern_map() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for key, pattern in STEP_TO_PATTERN.items():
        for idx, line in enumerate(CODE_LINES):
            if pattern in line:
                mapping[key] = idx
                break
    return mapping


_STEP_TO_LINE = _build_pattern_map()


def find_line(step_name: str, step_kind: str) -> int:
    for key, line in _STEP_TO_LINE.items():
        if key in step_kind or key in step_name:
            return line
    return -1


_HIGHLIGHT_COLOR = QColor("#264F78")
_HIGHLIGHT_BORDER_COLOR = QColor("#00FFFF")


class LineNumberArea(QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self._editor = editor

    def paintEvent(self, event) -> None:
        self._editor._paint_line_numbers(event)

    def sizeHint(self):
        return QSize(self._editor._line_number_area_width(), 0)


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet(
            "QPlainTextEdit { background-color: #0B0B0B; color: #D4D4D4; border: none; }"
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setReadOnly(True)
        self.setTabStopDistance(28)
        self.setFrameStyle(0)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self._highlighted_row: int = -1
        self._tooltips: dict[int, str] = {}

        self._line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._highlight_current_line()
        self._update_line_number_area_width()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        width = self._line_number_area_width()
        self._line_number_area.setGeometry(cr.left(), cr.top(), width, cr.height())

    def set_tooltips(self, tips: dict[int, str]) -> None:
        self._tooltips = tips

    def highlight_row(self, row: int) -> None:
        self._highlighted_row = row
        if row >= 0:
            cursor = QTextCursor(self.document().findBlockByNumber(row))
            self.setTextCursor(cursor)
            self._apply_extra_selection(row)
            self.centerCursor()
        else:
            self._highlight_current_line()

    def _apply_extra_selection(self, row: int) -> None:
        block = self.document().findBlockByNumber(row)
        if not block.isValid():
            return
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(_HIGHLIGHT_COLOR)
        sel.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        sel.cursor = QTextCursor(block)
        sel.cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        sel.cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        self.setExtraSelections([sel])

    def _paint_line_numbers(self, event) -> None:
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QColor("#1E1E1E"))
        painter.setFont(self.font())

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                is_hl = block_number == self._highlighted_row
                painter.setPen(QColor("#88FFFF" if is_hl else "#666666"))
                if is_hl:
                    painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
                else:
                    painter.setFont(QFont("Consolas", 11))
                painter.drawText(
                    0, top,
                    self._line_number_area.width() - 8, self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(block_number + 1),
                )

            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def _line_number_area_width(self):
        digits = len(str(max(1, self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_number_area_width(self) -> None:
        width = self._line_number_area_width()
        self.setViewportMargins(width, 0, 0, 0)

    def _update_line_number_area(self, rect, dy) -> None:
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width()

    def _highlight_current_line(self) -> None:
        if self._highlighted_row >= 0:
            return
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(QColor("#2A2D2E"))
        sel.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        cursor = self.cursorForPosition(event.pos())
        block = cursor.block()
        line = block.blockNumber()
        tip = self._tooltips.get(line, "")
        if tip:
            self.setToolTip(tip)
        else:
            self.setToolTip("")


class CodeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.result: TraceResult | None = None
        self._controller: StepRevealController | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.editor = CodeEditor()
        self.editor.setPlainText(CODE_SOURCE)
        self.editor.set_tooltips(LINE_TOOLTIPS)
        self._highlighter = PythonSyntaxHighlighter(self.editor.document())
        left_layout.addWidget(self.editor)
        splitter.addWidget(left_panel)

        self.explanation = ExplanationPanel()
        splitter.addWidget(self.explanation)

        splitter.setSizes([800, 200])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #333333; }"
        )

        layout.addWidget(splitter, 1)

        self.original_font = QFont("Consolas", 11)
        self.original_font.setPointSize(11)

    def _apply_highlight(self, row: int) -> None:
        self.editor.highlight_row(row)

    def _style_button(self, btn: QPushButton) -> None:
        btn.setStyleSheet(
            "QPushButton { background-color: #007ACC; color: white; padding: 6px 16px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0098FF; }"
            "QPushButton:disabled { background-color: #555555; color: #AAAAAA; }"
        )

    def reset(self) -> None:
        if self._controller:
            self._controller.stop()
        self._controller = None
        self.result = None
        self.play_btn.setEnabled(False)
        self.skip_end_btn.setEnabled(False)
        self.step_label.setText("")
        self.editor.highlight_row(-1)
        self.explanation.clear()

    def set_result(self, result: TraceResult) -> None:
        self.result = result
        n = len(result.steps)
        self._controller = StepRevealController(n, delay_ms=self._delay_from_slider())
        self._controller.step_revealed.connect(self._on_step_revealed)
        self._controller.finished.connect(self._on_reveal_finished)
        self._controller.started.connect(self._on_reveal_started)
        self._controller.paused.connect(self._on_reveal_paused)
        self._controller.resumed.connect(self._on_reveal_resumed)
        self.play_btn.setEnabled(True)
        self.skip_end_btn.setEnabled(True)
        self.step_label.setText(f"0 / {n} passos")
        self._update_play_button_text()

    def _delay_from_slider(self) -> int:
        return int(1000 / max(1, self.speed_slider.value()))

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

    def _on_reveal_started(self, total: int) -> None:
        self.step_label.setText(f"0 / {total} passos")
        self._update_play_button_text()

    def _on_reveal_finished(self) -> None:
        self._update_play_button_text()

    def _on_reveal_paused(self) -> None:
        self._update_play_button_text()

    def _on_reveal_resumed(self) -> None:
        self._update_play_button_text()

    def _on_step_revealed(self, index: int) -> None:
        if self.result is None or index < 0 or index >= len(self.result.steps):
            return
        step = self.result.steps[index]
        code_line = find_line(step.name, step.kind)
        if code_line < 0:
            return
        self._apply_highlight(code_line)
        self.step_label.setText(f"{index + 1} / {len(self.result.steps)} passos")
        self.explanation.show_step(step, code_line)
