from __future__ import annotations

from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = []

        keyword_fmt = QTextCharFormat()
        keyword_fmt.setForeground(QColor("#569CD6"))
        keyword_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = [
            "import", "from", "for", "in", "def", "return", "if", "else", "elif",
            "not", "and", "or", "True", "False", "None", "with", "as", "class",
            "while", "try", "except", "finally", "raise", "pass", "break", "continue",
            "yield", "lambda", "assert",
        ]
        for kw in keywords:
            self._rules.append((
                QRegularExpression(f"\\b{kw}\\b"),
                keyword_fmt,
            ))

        builtin_fmt = QTextCharFormat()
        builtin_fmt.setForeground(QColor("#DCDAA8"))
        for bn in ["enumerate", "range", "len", "int", "float", "print", "zip", "list", "dict", "str", "type", "isinstance", "super", "open", "hasattr", "getattr", "setattr"]:
            self._rules.append((
                QRegularExpression(f"\\b{bn}\\b"),
                builtin_fmt,
            ))

        self._string_fmt = QTextCharFormat()
        self._string_fmt.setForeground(QColor("#CE9178"))

        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#6A9955"))
        self._comment_fmt.setFontItalic(True)

        self._decorator_fmt = QTextCharFormat()
        self._decorator_fmt.setForeground(QColor("#D7BA7D"))

        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#B5CEA8"))

        self._triple_region: list[tuple[QRegularExpression, QRegularExpression, QTextCharFormat]] = []

        triple_str_fmt = QTextCharFormat()
        triple_str_fmt.setForeground(QColor("#CE9178"))
        self._triple_region.append((
            QRegularExpression("\"\"\""),
            QRegularExpression("\"\"\""),
            triple_str_fmt,
        ))
        self._triple_region.append((
            QRegularExpression("'''"),
            QRegularExpression("'''"),
            triple_str_fmt,
        ))

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)

        for start_pattern, end_pattern, fmt in self._triple_region:
            self._match_multiline(text, start_pattern, end_pattern, fmt)

        self._highlight_strings(text)
        self._highlight_comments(text)
        self._highlight_numbers(text)
        self._highlight_decorators(text)

    def _match_multiline(self, text: str, start_pattern, end_pattern, fmt) -> None:
        prev_state = self.previousBlockState()
        start_idx = 0
        if prev_state != 1:
            match = start_pattern.match(text)
            if not match.hasMatch():
                self.setCurrentBlockState(0)
                return
            start_idx = match.capturedStart() + match.capturedLength()
            self.setFormat(match.capturedStart(), match.capturedLength(), fmt)
        else:
            start_idx = 0

        while start_idx < len(text):
            end_match = end_pattern.match(text, start_idx)
            if not end_match.hasMatch():
                self.setFormat(start_idx, len(text) - start_idx, fmt)
                self.setCurrentBlockState(1)
                return
            end_idx = end_match.capturedStart() + end_match.capturedLength()
            self.setFormat(start_idx, end_idx - start_idx, fmt)
            start_idx = end_idx
            next_start = start_pattern.match(text, start_idx)
            if next_start.hasMatch():
                start_idx = next_start.capturedStart() + next_start.capturedLength()
                self.setFormat(next_start.capturedStart(), next_start.capturedLength(), fmt)

        self.setCurrentBlockState(0)

    def _highlight_strings(self, text: str) -> None:
        i = 0
        while i < len(text):
            if text[i] in ('"', "'"):
                if text[i:i+3] == text[i]*3:
                    i += 3
                    continue
                quote = text[i]
                start = i
                i += 1
                while i < len(text):
                    if text[i] == "\\":
                        i += 2
                        continue
                    if text[i] == quote:
                        i += 1
                        break
                    i += 1
                self.setFormat(start, i - start, self._string_fmt)
            else:
                i += 1

    def _highlight_comments(self, text: str) -> None:
        idx = text.find("#")
        if idx >= 0:
            in_string = False
            for j in range(idx):
                if text[j] in ('"', "'"):
                    if text[j:j+3] == text[j]*3:
                        pass
                    elif not in_string:
                        in_string = True
                    else:
                        in_string = False
            if not in_string:
                self.setFormat(idx, len(text) - idx, self._comment_fmt)

    def _highlight_numbers(self, text: str) -> None:
        it = QRegularExpression("\\b\\d+\\.?\\d*\\b").globalMatch(text)
        while it.hasNext():
            match = it.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self._number_fmt)

    def _highlight_decorators(self, text: str) -> None:
        it = QRegularExpression("^\\s*@\\w+").globalMatch(text)
        while it.hasNext():
            match = it.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self._decorator_fmt)
