from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from PyQt6.QtCore import QThread, pyqtSignal

from gpt2_trace_viewer.application.trace_result import (
    AttentionRecord,
    TopToken,
    TraceResult,
)
from gpt2_trace_viewer.domain.trace_step import TraceStep


class RealForwardTraceWorker(QThread):
    """Runs a manual GPT-2 forward pass and records real intermediate tensors."""

    progress = pyqtSignal(str, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, prompt: str, model: object, tokenizer: object) -> None:
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.tokenizer = tokenizer

    def run(self) -> None:
        try:
            result = self._trace()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _trace(self) -> TraceResult:
        model = self.model
        tokenizer = self.tokenizer
        self.progress.emit("Tokenizando prompt...", 0)

        steps: list[TraceStep] = []
        attention_records: list[AttentionRecord] = []

        def add_step(
            name: str,
            kind: str,
            tensor: torch.Tensor,
            block: int | None = None,
            head: int | None = None,
            description: str = "",
        ) -> None:
            steps.append(
                TraceStep(
                    name=name,
                    kind=kind,
                    tensor=tensor,
                    block=block,
                    head=head,
                    description=description,
                )
            )

        with torch.no_grad():
            input_ids = tokenizer.encode(self.prompt, return_tensors="pt")
            tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()]
            seq_len = input_ids.shape[1]

            add_step(
                name="input_ids",
                kind="Token IDs",
                tensor=input_ids,
                description="IDs produzidos pelo tokenizador.",
            )

            position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

            add_step(
                name="position_ids",
                kind="Position IDs",
                tensor=position_ids,
                description="Índices posicionais usados pelo GPT-2.",
            )

            token_embeddings = model.transformer.wte(input_ids)
            position_embeddings = model.transformer.wpe(position_ids)

            add_step(
                name="transformer.wte(input_ids)",
                kind="Token Embedding",
                tensor=token_embeddings,
                description="Embedding aprendido dos tokens.",
            )
            add_step(
                name="transformer.wpe(position_ids)",
                kind="Position Embedding",
                tensor=position_embeddings,
                description="Embedding aprendido das posições.",
            )

            hidden_states = token_embeddings + position_embeddings
            add_step(
                name="hidden_states = token_embeddings + position_embeddings",
                kind="Embedding Sum",
                tensor=hidden_states,
                description="Soma do embedding de token com o embedding posicional.",
            )

            embed_dim = model.config.n_embd
            num_heads = model.config.n_head
            head_dim = embed_dim // num_heads

            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(
                1, 1, seq_len, seq_len
            )

            for block_index, block in enumerate(model.transformer.h):
                self.progress.emit(f"Processando bloco {block_index + 1}/12...", block_index + 1)
                residual_attention = hidden_states
                add_step(
                    name=f"block_{block_index}.residual_attention_input",
                    kind="Residual Input",
                    tensor=residual_attention,
                    block=block_index,
                    description="Entrada preservada para soma residual depois da atenção.",
                )

                ln_1_out = block.ln_1(hidden_states)
                add_step(
                    name=f"block_{block_index}.ln_1",
                    kind="LayerNorm",
                    tensor=ln_1_out,
                    block=block_index,
                    description="Normalização antes da atenção.",
                )

                qkv = (ln_1_out @ block.attn.c_attn.weight) + block.attn.c_attn.bias
                add_step(
                    name=f"block_{block_index}.attn.c_attn",
                    kind="Linear QKV fused",
                    tensor=qkv,
                    block=block_index,
                    description="Projeção fundida que gera Q, K e V juntos.",
                )

                q, k, v = qkv.split(embed_dim, dim=-1)
                add_step(
                    name=f"block_{block_index}.q",
                    kind="Q split",
                    tensor=q,
                    block=block_index,
                    description="Parte Q extraída da projeção fundida.",
                )
                add_step(
                    name=f"block_{block_index}.k",
                    kind="K split",
                    tensor=k,
                    block=block_index,
                    description="Parte K extraída da projeção fundida.",
                )
                add_step(
                    name=f"block_{block_index}.v",
                    kind="V split",
                    tensor=v,
                    block=block_index,
                    description="Parte V extraída da projeção fundida.",
                )

                q_heads = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                k_heads = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                v_heads = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

                add_step(
                    name=f"block_{block_index}.q_heads",
                    kind="Q heads",
                    tensor=q_heads,
                    block=block_index,
                    description="Q separado em múltiplas cabeças.",
                )
                add_step(
                    name=f"block_{block_index}.k_heads",
                    kind="K heads",
                    tensor=k_heads,
                    block=block_index,
                    description="K separado em múltiplas cabeças.",
                )
                add_step(
                    name=f"block_{block_index}.v_heads",
                    kind="V heads",
                    tensor=v_heads,
                    block=block_index,
                    description="V separado em múltiplas cabeças.",
                )

                scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
                add_step(
                    name=f"block_{block_index}.attention_scores",
                    kind="QK scores",
                    tensor=scores,
                    block=block_index,
                    description="Scores brutos de atenção antes da máscara causal.",
                )

                masked_scores = scores.masked_fill(causal_mask == 0, float("-inf"))
                display_masked_scores = masked_scores.masked_fill(
                    torch.isinf(masked_scores), 0.0
                )
                add_step(
                    name=f"block_{block_index}.masked_attention_scores",
                    kind="Causal Mask",
                    tensor=display_masked_scores,
                    block=block_index,
                    description=(
                        "Scores após a máscara causal. Valores -inf são mostrados como 0 "
                        "apenas para estatísticas e visualização."
                    ),
                )

                attention_probs = F.softmax(masked_scores, dim=-1)
                add_step(
                    name=f"block_{block_index}.attention_probs",
                    kind="Attention Softmax",
                    tensor=attention_probs,
                    block=block_index,
                    description="Distribuição real de atenção após softmax.",
                )

                for head_index in range(num_heads):
                    attention_records.append(
                        AttentionRecord(
                            block=block_index,
                            head=head_index,
                            matrix=attention_probs[0, head_index].detach().cpu(),
                        )
                    )

                context_heads = attention_probs @ v_heads

                for head_index in range(num_heads):
                    add_step(
                        name=f"block_{block_index}.head_{head_index}.context",
                        kind=f"Attn Head {head_index}",
                        tensor=context_heads[:, head_index:head_index + 1],
                        block=block_index,
                        head=head_index,
                        description=f"Cabeca {head_index} do bloco {block_index}: resultado real da atencao.",
                    )

                merged_context = (
                    context_heads.transpose(1, 2)
                    .contiguous()
                    .view(1, seq_len, embed_dim)
                )
                add_step(
                    name=f"block_{block_index}.merged_heads",
                    kind="Merge Heads",
                    tensor=merged_context,
                    block=block_index,
                    description="Concatenação das cabeças de atenção.",
                )

                attention_projection = (
                    merged_context @ block.attn.c_proj.weight
                ) + block.attn.c_proj.bias
                add_step(
                    name=f"block_{block_index}.attn.c_proj",
                    kind="Attention Output Projection",
                    tensor=attention_projection,
                    block=block_index,
                    description="Projeção final da atenção.",
                )

                hidden_states = residual_attention + attention_projection
                add_step(
                    name=f"block_{block_index}.residual_after_attention",
                    kind="Residual Add",
                    tensor=hidden_states,
                    block=block_index,
                    description="Soma residual: entrada do bloco + saída da atenção.",
                )

                residual_mlp = hidden_states
                add_step(
                    name=f"block_{block_index}.residual_mlp_input",
                    kind="Residual Input",
                    tensor=residual_mlp,
                    block=block_index,
                    description="Entrada preservada para soma residual depois do MLP.",
                )

                ln_2_out = block.ln_2(hidden_states)
                add_step(
                    name=f"block_{block_index}.ln_2",
                    kind="LayerNorm",
                    tensor=ln_2_out,
                    block=block_index,
                    description="Normalização antes do MLP.",
                )

                mlp_fc = (ln_2_out @ block.mlp.c_fc.weight) + block.mlp.c_fc.bias
                add_step(
                    name=f"block_{block_index}.mlp.c_fc",
                    kind="MLP Expand",
                    tensor=mlp_fc,
                    block=block_index,
                    description="Expansão do MLP: 768 → 3072.",
                )

                gelu = F.gelu(mlp_fc)
                add_step(
                    name=f"block_{block_index}.mlp.gelu",
                    kind="GELU",
                    tensor=gelu,
                    block=block_index,
                    description="Ativação GELU real aplicada ao MLP.",
                )

                mlp_projection = (gelu @ block.mlp.c_proj.weight) + block.mlp.c_proj.bias
                add_step(
                    name=f"block_{block_index}.mlp.c_proj",
                    kind="MLP Project",
                    tensor=mlp_projection,
                    block=block_index,
                    description="Projeção do MLP: 3072 → 768.",
                )

                hidden_states = residual_mlp + mlp_projection
                add_step(
                    name=f"block_{block_index}.residual_after_mlp",
                    kind="Residual Add",
                    tensor=hidden_states,
                    block=block_index,
                    description="Soma residual: entrada do MLP + saída do MLP.",
                )

            final_norm = model.transformer.ln_f(hidden_states)
            add_step(
                name="transformer.ln_f",
                kind="Final LayerNorm",
                tensor=final_norm,
                description="Normalização final do transformer.",
            )

            last_token_state = final_norm[:, -1, :]
            add_step(
                name="last_token_state",
                kind="Last Token State",
                tensor=last_token_state,
                description="Estado oculto do último token usado para prever o próximo token.",
            )

            logits = last_token_state @ model.lm_head.weight.T
            add_step(
                name="lm_head",
                kind="Vocabulary Projection",
                tensor=logits,
                description="Projeção para o vocabulário.",
            )

            probabilities = F.softmax(logits, dim=-1)
            add_step(
                name="softmax_vocab",
                kind="Vocabulary Softmax",
                tensor=probabilities,
                description="Probabilidades reais para o próximo token.",
            )

            top_probs, top_ids = torch.topk(probabilities[0], 10)
            top_tokens = [
                TopToken(
                    rank=rank,
                    token_id=int(token_id.item()),
                    token=tokenizer.decode([int(token_id.item())]),
                    probability=float(prob.item()),
                )
                for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), start=1)
            ]

            result_text = self.prompt + tokenizer.decode([int(top_ids[0].item())])

        return TraceResult(
            prompt=self.prompt,
            result_text=result_text,
            tokens=tokens,
            steps=steps,
            attention_records=attention_records,
            top_tokens=top_tokens,
        )
