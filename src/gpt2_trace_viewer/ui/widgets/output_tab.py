from __future__ import annotations

import html

from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from gpt2_trace_viewer.application.trace_result import TraceResult


class OutputTab(QWidget):
    """Displays next-token probabilities and final generated text."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.console = QTextBrowser()
        self.console.setStyleSheet(
            "background-color: #0B0B0B; color: #DDDDDD; font-family: Consolas;"
        )
        layout.addWidget(self.console)

    def reset(self) -> None:
        self.console.clear()

    def set_result(self, result: TraceResult) -> None:
        rows_html = ""
        for token in result.top_tokens:
            display_token = html.escape(token.token).replace(" ", "␣").replace("\n", "\\n")
            rows_html += f"""
            <tr>
                <td>{token.rank}</td>
                <td>{token.token_id}</td>
                <td><b>{display_token}</b></td>
                <td>{token.probability * 100:.4f}%</td>
            </tr>
            """

        prompt = html.escape(result.prompt)
        result_text = html.escape(result.result_text)

        self.console.setHtml(
            f"""
            <h2 style="color:#00FF00;">Resultado</h2>
            <p><b>Prompt:</b> {prompt}</p>
            <p><b>Próximo texto:</b> {result_text}</p>
            <h3 style="color:#FFD700;">Top tokens reais</h3>
            <table border="1" cellspacing="0" cellpadding="5">
                <tr>
                    <th>Rank</th>
                    <th>Token ID</th>
                    <th>Token</th>
                    <th>Probabilidade</th>
                </tr>
                {rows_html}
            </table>
            """
        )
