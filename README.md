# GPT-2 Real Trace Viewer

Projeto PyQt6 para visualizar o **forward pass real** do GPT-2, sem animação fake, sem `random`, sem pulso hardcoded.

Ele mostra:

- fluxo completo do GPT-2;
- grafo neural real por etapa;
- shapes, média, desvio, norma L2, min/max;
- top ativações por tensor;
- atenção real por bloco e head;
- top tokens reais do próximo token.

## Estrutura

```text
gpt2-real-trace-viewer/
├── main.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── src/
    └── gpt2_trace_viewer/
        ├── app.py
        ├── __main__.py
        ├── application/
        │   ├── forward_tracer.py
        │   └── trace_result.py
        ├── domain/
        │   ├── tensor_inspector.py
        │   └── trace_step.py
        ├── infra/
        │   └── model_loader.py
        └── ui/
            ├── main_window.py
            └── widgets/
                ├── attention_matrix_viewer.py
                ├── attention_tab.py
                ├── graph_viewer.py
                ├── output_tab.py
                └── trace_tab.py
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -U pip
pip install -r requirements.txt
```

## Rodar direto

```bash
python main.py
```

## Rodar como pacote editável

```bash
pip install -e .
gpt2-trace-viewer
```

## Observações

Na primeira execução, o `transformers` baixa o modelo `gpt2` para o cache local.

A aba de grafo não desenha feixes aleatórios. Cada nó representa uma etapa real do forward pass, e a espessura das setas é calculada a partir da norma L2 real do tensor de destino.
