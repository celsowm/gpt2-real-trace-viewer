from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class SplashScreen(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setFixedSize(500, 200)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        title = QLabel("GPT-2 Real Trace Viewer")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.message = QLabel("Inicializando…")
        self.message.setFont(QFont("Arial", 11))
        self.message.setStyleSheet("color: #AAAAAA;")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(
            "QProgressBar { background-color: #333333; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #00A86B; border-radius: 3px; }"
        )
        layout.addWidget(self.progress)

        self._center_on_screen()

    def _center_on_screen(self) -> None:
        from PyQt6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = (geo.width() - self.width()) // 2
        y = (geo.height() - self.height()) // 2
        self.move(x, y)

    def set_message(self, text: str) -> None:
        self.message.setText(text)
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()
