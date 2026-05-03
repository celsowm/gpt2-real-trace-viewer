from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gpt2_trace_viewer.application.forward_tracer import RealForwardTraceWorker
from gpt2_trace_viewer.application.trace_result import TraceResult
from gpt2_trace_viewer.infra.model_loader import ModelLoaderThread
from gpt2_trace_viewer.ui.widgets.attention_tab import AttentionTab
from gpt2_trace_viewer.ui.widgets.code_tab import CodeTab
from gpt2_trace_viewer.ui.widgets.graph_viewer import RealNeuralGraphViewer
from gpt2_trace_viewer.ui.widgets.output_tab import OutputTab
from gpt2_trace_viewer.ui.widgets.trace_tab import TraceTab


class MainWindow(QMainWindow):
    """Application shell. It wires UI widgets to loader/tracer workers."""

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.loader_thread: ModelLoaderThread | None = None
        self.trace_worker: RealForwardTraceWorker | None = None

        self.setWindowTitle("GPT-2 Real Forward Trace Viewer")
        self.resize(1350, 900)
        self.setStyleSheet("background-color: #1E1E1E; color: #FFFFFF;")

        self._build_ui()
        self._start_model_loading()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        top_layout = QHBoxLayout()

        self.status_label = QLabel("Carregando GPT-2...")
        self.status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.status_label.setStyleSheet("color: #FFD700;")
        top_layout.addWidget(self.status_label, 2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setStyleSheet("QProgressBar::chunk { background-color: #00A86B; }")
        top_layout.addWidget(self.progress, 1)

        root_layout.addLayout(top_layout)

        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))

        self.prompt_input = QLineEdit("We the people of")
        self.prompt_input.setFont(QFont("Consolas", 12))
        self.prompt_input.setEnabled(False)
        self.prompt_input.setStyleSheet(
            "background-color: #2D2D30; border: 1px solid #555; padding: 6px; color: white;"
        )
        prompt_layout.addWidget(self.prompt_input, 1)

        self.run_button = QPushButton("Executar forward real")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_trace)
        self.run_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007ACC;
                color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098FF;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
            """
        )
        prompt_layout.addWidget(self.run_button)

        root_layout.addLayout(prompt_layout)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #444;
            }
            QTabBar::tab {
                background: #2D2D30;
                color: #AAAAAA;
                padding: 10px 20px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #007ACC;
                color: white;
            }
            """
        )
        root_layout.addWidget(self.tabs, 1)

        self.trace_tab = TraceTab()
        self.graph_tab = RealNeuralGraphViewer()
        self.attention_tab = AttentionTab()
        self.output_tab = OutputTab()
        self.code_tab = CodeTab()

        self.tabs.addTab(self.trace_tab, "Forward real")
        self.tabs.addTab(self.graph_tab, "Grafo neural real")
        self.tabs.addTab(self.code_tab, "Codigo torch")
        self.tabs.addTab(self.attention_tab, "Atencao real")
        self.tabs.addTab(self.output_tab, "Output")

    def _start_model_loading(self) -> None:
        self.loader_thread = ModelLoaderThread("gpt2")
        self.loader_thread.loaded.connect(self._on_model_loaded)
        self.loader_thread.failed.connect(self._on_model_failed)
        self.loader_thread.start()

    def _on_model_loaded(self, model: object, tokenizer: object) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.status_label.setText("GPT-2 carregado. Pronto para executar forward real.")
        self.status_label.setStyleSheet("color: #00FF00;")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        self.prompt_input.setEnabled(True)
        self.run_button.setEnabled(True)

    def _on_model_failed(self, message: str) -> None:
        self.status_label.setText(f"Erro ao carregar GPT-2: {message}")
        self.status_label.setStyleSheet("color: #FF5555;")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def _run_trace(self) -> None:
        if self.model is None or self.tokenizer is None:
            return

        prompt = self.prompt_input.text().strip()
        if not prompt:
            return

        self._set_running_state(True)
        self._reset_result_widgets()

        self.trace_worker = RealForwardTraceWorker(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.trace_worker.progress.connect(self._on_trace_progress)
        self.trace_worker.finished.connect(self._on_trace_finished)
        self.trace_worker.failed.connect(self._on_trace_failed)
        self.trace_worker.start()

    def _set_running_state(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.prompt_input.setEnabled(not running)

        if running:
            self.status_label.setText("Processando forward real...")
            self.status_label.setStyleSheet("color: #FFD700;")
            self.progress.setRange(0, 12)
            self.progress.setValue(0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)

    def _reset_result_widgets(self) -> None:
        self.trace_tab.reset()
        self.graph_tab.reset()
        self.code_tab.reset()
        self.attention_tab.reset()
        self.output_tab.reset()

    def _on_trace_progress(self, message: str, value: int) -> None:
        self.status_label.setText(message)
        if 0 <= value <= 12:
            self.progress.setValue(value)

    def _on_trace_finished(self, result: TraceResult) -> None:
        self.trace_tab.set_result(result)
        self.graph_tab.set_result(result)
        self.code_tab.set_result(result)
        self.attention_tab.set_result(result)
        self.output_tab.set_result(result)

        self.prompt_input.setText(result.result_text)
        self.status_label.setText("Forward real concluído.")
        self.status_label.setStyleSheet("color: #00FF00;")
        self._set_running_state(False)

    def _on_trace_failed(self, message: str) -> None:
        self.status_label.setText(f"Erro no forward real: {message}")
        self.status_label.setStyleSheet("color: #FF5555;")
        self._set_running_state(False)
