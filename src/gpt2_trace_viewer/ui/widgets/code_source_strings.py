import os

CODE_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "gpt2_real.py")
with open(CODE_SOURCE_PATH, "r") as f:
    CODE_SOURCE = f.read()