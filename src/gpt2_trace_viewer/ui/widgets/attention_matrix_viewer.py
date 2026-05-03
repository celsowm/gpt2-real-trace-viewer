from __future__ import annotations

from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView


class AttentionMatrixViewer(QGraphicsView):
    """Draws a real attention matrix for a selected block/head."""

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#111111"))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def clear(self) -> None:
        self.scene.clear()

    def draw_attention(self, tokens: list[str], matrix: list[list[float]]) -> None:
        self.scene.clear()

        cell_size = 58
        offset_x = 140
        offset_y = 120

        font = QFont("Consolas", 9)
        title_font = QFont("Arial", 13, QFont.Weight.Bold)

        title = self.scene.addText("Matriz de atenção real: softmax(QKᵀ / sqrt(d) + mask)")
        title.setFont(title_font)
        title.setDefaultTextColor(QColor("#FFD700"))
        title.setPos(20, 20)

        clean_tokens = [
            token.replace("\n", "\\n").replace(" ", "␣")
            for token in tokens
        ]

        for i, token in enumerate(clean_tokens):
            row_label = self.scene.addText(f"Q {i}: {token}")
            row_label.setFont(font)
            row_label.setDefaultTextColor(QColor("#FFD700"))
            row_label.setPos(15, offset_y + i * cell_size + 18)

            col_label = self.scene.addText(f"K {i}: {token}")
            col_label.setFont(font)
            col_label.setDefaultTextColor(QColor("#00FFFF"))
            col_label.setRotation(-35)
            col_label.setPos(offset_x + i * cell_size + 8, 80)

        for row_index, row in enumerate(matrix):
            for col_index, raw_value in enumerate(row):
                value = float(raw_value)
                intensity = max(0, min(255, int(value * 255 * 2.5)))

                if value <= 0.00001:
                    color = QColor("#181818")
                    text_color = QColor("#777777")
                else:
                    color = QColor(0, intensity, 255)
                    text_color = QColor("black") if intensity > 160 else QColor("white")

                self.scene.addRect(
                    offset_x + col_index * cell_size,
                    offset_y + row_index * cell_size,
                    cell_size,
                    cell_size,
                    QPen(QColor("#333333")),
                    QBrush(color),
                )

                if value > 0.001:
                    value_text = self.scene.addText(f"{value:.2f}")
                    value_text.setFont(QFont("Consolas", 8))
                    value_text.setDefaultTextColor(text_color)
                    value_text.setPos(
                        offset_x + col_index * cell_size + 12,
                        offset_y + row_index * cell_size + 18,
                    )
