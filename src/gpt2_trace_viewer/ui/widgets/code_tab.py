from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QTextCursor
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

CODE_SOURCE = r'''import torch
import torch.nn.functional as F

# === TOKENIZACAO ===
input_ids = tokenizer.encode(prompt, return_tensors="pt")
tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
seq_len = input_ids.shape[1]
position_ids = torch.arange(0, seq_len).unsqueeze(0)

# === EMBEDDING ===
token_embeddings = model.transformer.wte(input_ids)
position_embeddings = model.transformer.wpe(position_ids)
hidden_states = token_embeddings + position_embeddings

# === CONFIG ===
embed_dim = model.config.n_embd
num_heads = model.config.n_head
head_dim = embed_dim // num_heads
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

# === BLOCO TRANSFORMER (x12) ===
for block_index, block in enumerate(model.transformer.h):

    # --- ATENCAO ---
    residual_attention = hidden_states

    ln_1_out = block.ln_1(hidden_states)
    qkv = ln_1_out @ block.attn.c_attn.weight + block.attn.c_attn.bias
    q, k, v = qkv.split(embed_dim, dim=-1)

    q_heads = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_heads = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    v_heads = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = (q_heads @ k_heads.transpose(-2, -1)) / (head_dim ** 0.5)
    masked_scores = scores.masked_fill(causal_mask == 0, float("-inf"))
    attention_probs = F.softmax(masked_scores, dim=-1)

    context_heads = attention_probs @ v_heads

    merged = context_heads.transpose(1, 2).contiguous().view(1, seq_len, embed_dim)
    attn_out = merged @ block.attn.c_proj.weight + block.attn.c_proj.bias
    hidden_states = residual_attention + attn_out

    # --- MLP ---
    residual_mlp = hidden_states

    ln_2_out = block.ln_2(hidden_states)
    mlp_fc = ln_2_out @ block.mlp.c_fc.weight + block.mlp.c_fc.bias
    gelu = F.gelu(mlp_fc)
    mlp_out = gelu @ block.mlp.c_proj.weight + block.mlp.c_proj.bias
    hidden_states = residual_mlp + mlp_out

# === FINAL ===
final_norm = model.transformer.ln_f(hidden_states)
last_token_state = final_norm[:, -1, :]
logits = last_token_state @ model.lm_head.weight.T
probabilities = F.softmax(logits, dim=-1)
top_probs, top_ids = torch.topk(probabilities[0], 10)
result_text = prompt + tokenizer.decode([top_ids[0]])
'''

LINE_TOOLTIPS: dict[int, str] = {
    1: "Importa o PyTorch, framework usado pelo GPT-2.",
    2: "Importa funcoes de ativacao como softmax e gelu.",
    5: "Converte o prompt em IDs numericos de tokens.",
    6: "Decodifica cada ID de volta para o texto do token.",
    7: "Numero de tokens na sequencia de entrada.",
    8: "Indices posicionais 0, 1, 2, ... para cada token.",
    11: "Embedding de tokens: mapeia cada ID para um vetor 768D.",
    12: "Embedding posicional: mapeia cada posicao para um vetor 768D.",
    13: "Soma os dois embeddings para formar o estado oculto inicial.",
    16: "Dimensao do embedding (768 para GPT-2 small).",
    17: "Numero de cabecas de atencao (12).",
    18: "Dimensao de cada cabeca (64 = 768 / 12).",
    19: "Mascara causal: impede tokens de verem tokens futuros.",
    22: "Loop sobre cada bloco transformer (0 a 11).",
    25: "Preserva o estado atual para a soma residual.",
    27: "Normalizacao LayerNorm antes da atencao.",
    28: "Projecao linear fundida: gera Q, K e V juntos (768 -> 2304).",
    29: "Divide o tensor fundido em Q (768), K (768) e V (768).",
    31: "Rearranja Q de (B, S, 768) para (B, 12, S, 64) por cabeca.",
    32: "Rearranja K de (B, S, 768) para (B, 12, S, 64) por cabeca.",
    33: "Rearranja V de (B, S, 768) para (B, 12, S, 64) por cabeca.",
    35: "Calcula scores de atencao: Q @ K.T / sqrt(64) para cada cabeca.",
    36: "Aplica a mascara causal: -inf para posicoes proibidas.",
    37: "Softmax: converte scores em probabilidades de atencao.",
    39: "Atencao aplicada: pondera V pelas probabilidades.",
    42: "Concatena as 12 cabecas de volta para (B, S, 768).",
    43: "Projecao linear de saida da atencao (768 -> 768).",
    44: "Soma residual: entrada + saida da atencao.",
    47: "Preserva o estado para a segunda soma residual.",
    49: "LayerNorm antes do MLP.",
    50: "Expansao do MLP: 768 -> 3072 (camada oculta larga).",
    51: "Ativacao GELU nao-linear aplicada a expansao.",
    52: "Projecao do MLP: 3072 -> 768 de volta.",
    53: "Soma residual: entrada do MLP + saida do MLP.",
    56: "Ultima LayerNorm apos todos os blocos.",
    57: "Pega o estado oculto do ultimo token apenas.",
    58: "Projeta o estado no vocabulario: 768 -> 50257 logits.",
    59: "Softmax sobre logits: probabilidades para cada token.",
    60: "Pega os 10 tokens mais provaveis.",
    61: "Concatena o prompt com o token mais provavel.",
}

CODE_LINES = CODE_SOURCE.split("\n")

BG_HIGHLIGHT = QColor("#264F78")

STEP_TO_CODE: dict[str, str] = {
    "input_ids": "input_ids = tokenizer",
    "position_ids": "position_ids = torch",
    "wte(input_ids)": "token_embeddings =",
    "wpe(position_ids)": "position_embeddings =",
    "Embedding Sum": "hidden_states = token_embeddings",
    "residual_attention_input": "residual_attention =",
    "residual_mlp_input": "residual_mlp =",
    "ln_1": "ln_1_out =",
    "ln_2": "ln_2_out =",
    "attn.c_attn": "qkv =",
    "c_attn": "qkv =",
    "q split": "q, k, v = qkv.split",
    "k split": "q, k, v = qkv.split",
    "v split": "q, k, v = qkv.split",
    "q_heads": "q_heads =",
    "k_heads": "k_heads =",
    "v_heads": "v_heads =",
    "QK scores": "scores =",
    "Causal Mask": "masked_scores =",
    "Attention Softmax": "attention_probs =",
    "Attn Head": "context_heads =",
    "Merge Heads": "merged =",
    "Attention Output Projection": "attn_out =",
    "Residual Add": "hidden_states = residual",
    "MLP Expand": "mlp_fc =",
    "GELU": "gelu =",
    "MLP Project": "mlp_out =",
    "Final LayerNorm": "final_norm =",
    "Last Token State": "last_token_state =",
    "Vocabulary Projection": "logits =",
    "Vocabulary Softmax": "probabilities =",
}


def _build_step_map() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for key, pattern in STEP_TO_CODE.items():
        for idx, line in enumerate(CODE_LINES):
            if pattern in line:
                mapping[key] = idx
                break
    return mapping


_STEP_TO_LINE = _build_step_map()


def _find_line(step_name: str, step_kind: str) -> int:
    for key, line in _STEP_TO_LINE.items():
        if key in step_name or key in step_kind:
            return line
    return -1


TOOLTIP_CSS = """
<style>
  .tooltip-line {
    position: relative;
    display: inline;
  }
  .tooltip-line:hover::after {
    content: attr(data-tip);
    position: absolute;
    left: 0;
    bottom: 20px;
    background: #2D2D2D;
    color: #F0F0F0;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
    white-space: nowrap;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #555;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    z-index: 9999;
    pointer-events: none;
  }
</style>
"""


KEYWORDS = {
    "import", "from", "for", "in", "def", "return", "if", "else",
    "not", "and", "or", "True", "False", "None", "with", "as",
}
BUILTINS = {"enumerate", "range", "len", "int", "float", "print", "zip", "list", "dict"}

_HIGHLIGHT_CACHE: dict[int, str] = {}


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
        if escaped[i] == '"' or escaped[i] == "'":
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
        code_line = _find_line(step.name, step.kind)
        if code_line < 0:
            return
        self._apply_highlight(code_line)
        self.step_label.setText(f"{index + 1} / {len(self.result.steps)} passos")
