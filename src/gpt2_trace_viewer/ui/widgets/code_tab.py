from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gpt2_trace_viewer.application.trace_result import TraceResult
from gpt2_trace_viewer.ui.step_reveal import StepRevealController
from gpt2_trace_viewer.ui.widgets.code_source_strings import CODE_SOURCE


LINE_TOOLTIPS: dict[int, str] = {
    3: "Guarda o estado para soma residual (atenção)",
    4: "LayerNorm antes da atenção (ln_1)",
    5: "Chama GPT2Attention.forward",
    17: "Guarda estado para soma residual (MLP)",
    34: "c_attn = Conv1D(3*embed_dim, embed_dim) fundido",
    42: "Q @ K.T com fator de escala 1/sqrt(head_dim)",
    44: "Soma a máscara causal (-inf nas posições proibidas)",
    45: "Softmax: normaliza para probabilidades de atenção",
    57: "Expansão MLP: embed_dim -> intermediate_size (4x)",
    87: "Token embedding via wte (vocab_size -> embed_dim)",
    91: "Calcula position_ids se não fornecido",
}

CODE_LINES = CODE_SOURCE.split("\n")

KEYWORDS = {
    "import", "from", "for", "in", "def", "return", "if", "else",
    "not", "and", "or", "True", "False", "None", "with", "as", "class",
}
BUILTINS = {"enumerate", "range", "len", "int", "float", "print", "zip", "list", "dict"}

_HIGHLIGHT_CACHE: dict[str, str] = {}


def _highlight_line(line: str) -> str:
    cached = _HIGHLIGHT_CACHE.get(line)
    if cached is not None:
        return cached
    escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts: list[str] = []
    i = 0
    while i < len(escaped):
        if escaped[i : i + 7] == "&lt;!--":
            end = escaped.find("--&gt;", i + 7)
            if end == -1:
                end = len(escaped)
            parts.append(f"<span style='color:#6A9955;font-style:italic'>{escaped[i:end + 6]}</span>")
            i = end + 6
            continue
        if escaped[i] == "#":
            parts.append(f"<span style='color:#6A9955;font-style:italic'>{escaped[i:]}</span>")
            break
        if escaped[i] in ('"', "'"):
            quote = escaped[i]
            end = i + 1
            while end < len(escaped) and escaped[end] != quote:
                if escaped[end] == "\\":
                    end += 1
                end += 1
            end += 1
            parts.append(f"<span style='color:#CE9178'>{escaped[i:end]}</span>")
            i = end
            continue
        if escaped[i].isalpha() or escaped[i] == "_":
            j = i
            while j < len(escaped) and (escaped[j].isalnum() or escaped[j] == "_"):
                j += 1
            word = escaped[i:j]
            if word in KEYWORDS:
                parts.append(f"<span style='color:#569CD6;font-weight:bold'>{word}</span>")
            elif word in BUILTINS:
                parts.append(f"<span style='color:#DCDAA8'>{word}</span>")
            else:
                parts.append(word)
            i = j
            continue
        parts.append(escaped[i])
        i += 1
    result = "".join(parts)
    _HIGHLIGHT_CACHE[line] = result
    return result


def _build_rich_html(highlight_line: int = -1) -> str:
    style = (
        "<style>"
        ".code-line { line-height: 1.5; padding: 1px 0; }"
        ".hl { background-color: #264F78; border-left: 4px solid #00FFFF; }"
        ".hl-num { color: #88FFFF; font-weight: bold; }"
        ".line-num { color: #666; text-align: right; display: inline-block; width: 3em; padding-right: 1.2em; user-select: none; }"
        "</style>"
    )
    parts: list[str] = []
    for idx, raw_line in enumerate(CODE_LINES):
        tip = LINE_TOOLTIPS.get(idx, "")
        tip_attr = f' title="{tip}"' if tip else ""
        num_cls = "line-num hl-num" if idx == highlight_line else "line-num"
        hl_cls = "code-line hl" if idx == highlight_line else "code-line"
        highlighted = _highlight_line(raw_line)
        parts.append(
            f'<div class="{hl_cls}"{tip_attr}>'
            f'<span class="{num_cls}">{idx + 1}</span>'
            f'{highlighted}'
            f'</div>'
        )
    return f"<!DOCTYPE HTML><html><body style='background:#0B0B0B; font-family: Consolas, monospace; font-size: 13px; margin: 8px;'>{style}{''.join(parts)}</body></html>"


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


class CodeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.result: TraceResult | None = None
        self._controller: StepRevealController | None = None
        self._highlighted_row: int = -1

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
        layout.addLayout(controls)

        self.browser = QTextBrowser()
        self.browser.setFont(QFont("Consolas", 11))
        self.browser.setStyleSheet(
            "QTextBrowser { background-color: #0B0B0B; color: #D4D4D4; border: none; }"
        )
        self.browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)
        layout.addWidget(self.browser)

        self._render_all()

    def _render_all(self) -> None:
        self.browser.setHtml(_build_rich_html())

    def _apply_highlight(self, row: int) -> None:
        self.browser.setHtml(_build_rich_html(highlight_line=row))
        if row >= 0:
            cursor = self.browser.textCursor()
            block = self.browser.document().findBlockByLineNumber(row)
            if block.isValid():
                cursor.setPosition(block.position())
                self.browser.setTextCursor(cursor)
                self.browser.ensureCursorVisible()

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
        self._highlighted_row = -1
        self.play_btn.setEnabled(False)
        self.skip_end_btn.setEnabled(False)
        self.step_label.setText("")
        self._render_all()

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
