import sys

from PyQt6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)

    # splash antes de qualquer import pesado
    from gpt2_trace_viewer.ui.splash import SplashScreen

    splash = SplashScreen()
    splash.show()
    app.processEvents()
    splash.set_message("Carregando interface…")

    from gpt2_trace_viewer.ui.main_window import MainWindow

    splash.set_message("Construindo janela principal…")

    window = MainWindow(splash=splash)

    # splash.finish() é chamado internamente quando o modelo carregar
    window.show()
    return app.exec()
