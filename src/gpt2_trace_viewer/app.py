import sys

from PyQt6.QtWidgets import QApplication

from gpt2_trace_viewer.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
