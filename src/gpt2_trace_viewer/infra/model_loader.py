from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ModelLoaderThread(QThread):
    """Loads GPT-2 outside the UI thread."""

    loaded = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, model_name: str = "gpt2") -> None:
        super().__init__()
        self.model_name = model_name

    def run(self) -> None:
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
            model.eval()
            model.to("cpu")
            self.loaded.emit(model, tokenizer)
        except Exception as exc:
            self.failed.emit(str(exc))
