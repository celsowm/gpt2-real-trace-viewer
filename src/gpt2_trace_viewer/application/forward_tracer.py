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
                description=(
                    "Índices inteiros representando cada token do prompt no vocabulário GPT-2 "
                    "(vocab_size=50257). Ex: 'We' → 263, 'the' → 262. "
                    "Shape: [1, seq_len] — batch=1, seq_len=número de tokens."
                ),
            )

            position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)

            add_step(
                name="position_ids",
                kind="Position IDs",
                tensor=position_ids,
                description=(
                    "Índices posicionais [0, 1, 2, ..., seq_len-1]. "
                    "GPT-2 não tem recorrência ou convolução — precisa da posição de cada token "
                    "para saber a ordem da sequência. Shape: [1, seq_len]."
                ),
            )

            token_embeddings = model.transformer.wte(input_ids)
            position_embeddings = model.transformer.wpe(position_ids)

            add_step(
                name="transformer.wte(input_ids)",
                kind="Token Embedding",
                tensor=token_embeddings,
                description=(
                    "Consulta a tabela de embedding de tokens (wte = word token embedding). "
                    "Cada token ID vira um vetor de 768 dimensões. "
                    "É uma matriz treinável de shape [50257, 768] — uma linha por palavra no vocabulário. "
                    "Shape saída: [1, seq_len, 768]."
                ),
            )
            add_step(
                name="transformer.wpe(position_ids)",
                kind="Position Embedding",
                tensor=position_embeddings,
                description=(
                    "Consulta a tabela de embedding posicional (wpe = word position embedding). "
                    "Cada posição [0..1023] tem um vetor aprendido de 768 dimensões. "
                    "Isso permite que o modelo saiba onde cada token está na sequência. "
                    "Shape: [1, seq_len, 768], igual ao token embedding."
                ),
            )

            hidden_states = token_embeddings + position_embeddings
            add_step(
                name="hidden_states = token_embeddings + position_embeddings",
                kind="Embedding Sum",
                tensor=hidden_states,
                description=(
                    "Soma elemento-a-elemento do embedding do token com o embedding posicional. "
                    "Resultado: um vetor de 768 dimensões para cada posição que codifica "
                    "tanto o significado do token quanto sua posição na frase. "
                    "Shape: [1, seq_len, 768]."
                ),
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
                    description=(
                        "Copia o hidden_states ANTES da atenção. Essa cópia será somada de volta "
                        "após a atenção (soma residual / skip connection). "
                        "Essencial para treinar redes profundas — evita o desvanecimento do gradiente. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 768]."
                    ),
                )

                ln_1_out = block.ln_1(hidden_states)
                add_step(
                    name=f"block_{block_index}.ln_1",
                    kind="LayerNorm",
                    tensor=ln_1_out,
                    block=block_index,
                    description=(
                        "Layer Normalization (ln_1) antes da atenção. "
                        "Normaliza as ativações para ter média ~0 e variância ~1, depois "
                        "aplica escala e bias aprendidos (γ e β). "
                        "Ajuda a estabilizar o treinamento e reduz a sensibilidade à escala dos inputs. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 768]."
                    ),
                )

                qkv = (ln_1_out @ block.attn.c_attn.weight) + block.attn.c_attn.bias
                add_step(
                    name=f"block_{block_index}.attn.c_attn",
                    kind="Linear QKV fused",
                    tensor=qkv,
                    block=block_index,
                    description=(
                        "Projeção linear fundida que gera Query, Key e Value simultaneamente. "
                        "c_attn é uma matriz [768, 2304] que mapeia 768 → 2304 dimensões "
                        "(3 × 768 para Q, K, V concatenados). "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 2304]."
                    ),
                )

                q, k, v = qkv.split(embed_dim, dim=-1)
                add_step(
                    name=f"block_{block_index}.q",
                    kind="Q split",
                    tensor=q,
                    block=block_index,
                    description=(
                        "Parte Query — primeiro terço da projeção fundida. Query pergunta: "
                        "'O quanto cada token deve prestar atenção nos outros?'. "
                        f"Shape: [1, seq_len, 768] — Bloco {block_index + 1}/12."
                    ),
                )
                add_step(
                    name=f"block_{block_index}.k",
                    kind="K split",
                    tensor=k,
                    block=block_index,
                    description=(
                        "Parte Key — segundo terço da projeção fundida. Key responde: "
                        "'O quanto cada token merece receber atenção?'. "
                        "O produto Q @ K.T produz os scores de atenção. "
                        f"Shape: [1, seq_len, 768] — Bloco {block_index + 1}/12."
                    ),
                )
                add_step(
                    name=f"block_{block_index}.v",
                    kind="V split",
                    tensor=v,
                    block=block_index,
                    description=(
                        "Parte Value — terceiro terço da projeção fundida. Value carrega "
                        "a informação real que será agregada. "
                        "A saída da atenção é a média ponderada dos Values, onde os pesos "
                        "vêm do softmax(Q @ K.T). "
                        f"Shape: [1, seq_len, 768] — Bloco {block_index + 1}/12."
                    ),
                )

                q_heads = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                k_heads = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                v_heads = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

                add_step(
                    name=f"block_{block_index}.q_heads",
                    kind="Q heads",
                    tensor=q_heads,
                    block=block_index,
                    description=(
                        "Query remodelada para múltiplas cabeças de atenção. "
                        "De [1, seq_len, 768] → [1, 12, seq_len, 64]. "
                        "GPT-2 usa 12 cabeças paralelas, cada uma opera em 64 dimensões. "
                        "Cada cabeça pode aprender um padrão de atenção diferente. "
                        f"Bloco {block_index + 1}/12."
                    ),
                )
                add_step(
                    name=f"block_{block_index}.k_heads",
                    kind="K heads",
                    tensor=k_heads,
                    block=block_index,
                    description=(
                        "Key remodelada para 12 cabeças. "
                        "Shape: [1, 12, seq_len, 64]. "
                        f"Bloco {block_index + 1}/12. "
                        "Junto com Q, cada cabeça calcula atenção independentemente."
                    ),
                )
                add_step(
                    name=f"block_{block_index}.v_heads",
                    kind="V heads",
                    tensor=v_heads,
                    block=block_index,
                    description=(
                        "Value remodelada para 12 cabeças. "
                        "Shape: [1, 12, seq_len, 64]. "
                        f"Bloco {block_index + 1}/12. "
                        "Cada cabeça produz sua própria saída ponderada."
                    ),
                )

                scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
                add_step(
                    name=f"block_{block_index}.attention_scores",
                    kind="QK scores",
                    tensor=scores,
                    block=block_index,
                    description=(
                        "Scores de atenção brutos = Q @ K.T / √d_k. "
                        "Produto escalar entre cada query e cada key — mede compatibilidade. "
                        "Divide por √64 = 8 para evitar que scores cresçam muito com a dimensão. "
                        "Shape: [1, 12, seq_len, seq_len]. "
                        f"Bloco {block_index + 1}/12. Valores altos = mais atenção."
                    ),
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
                        "Máscara causal aplicada: tokens futuros viram -inf (depois mostrados como 0 "
                        "para visualização). "
                        "Isso garante que o token na posição i só pode ver tokens ≤ i. "
                        "Sem isso, o modelo 'trapacearia' vendo tokens futuros. "
                        f"Bloco {block_index + 1}/12. Forma: triângulo inferior."
                    ),
                )

                attention_probs = F.softmax(masked_scores, dim=-1)
                add_step(
                    name=f"block_{block_index}.attention_probs",
                    kind="Attention Softmax",
                    tensor=attention_probs,
                    block=block_index,
                    description=(
                        "Softmax sobre a última dimensão transforma scores em probabilidades "
                        "(valores entre 0 e 1, somam 1 por linha). "
                        "Tokens com -inf viram 0 (não recebem atenção). "
                        f"Bloco {block_index + 1}/12. "
                        "Cada linha i diz: 'quanto o token i deve prestar atenção em cada token j ≤ i'."
                    ),
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
                    description=(
                        "Concatena as 12 cabeças de volta: [1, 12, seq_len, 64] → [1, seq_len, 768]. "
                        "Cada cabeça contribui com 64 dimensões; juntas formam o vetor completo de 768. "
                        f"Bloco {block_index + 1}/12."
                    ),
                )

                attention_projection = (
                    merged_context @ block.attn.c_proj.weight
                ) + block.attn.c_proj.bias
                add_step(
                    name=f"block_{block_index}.attn.c_proj",
                    kind="Attention Output Projection",
                    tensor=attention_projection,
                    block=block_index,
                    description=(
                        "Projeção de saída da atenção (c_proj). "
                        "Uma transformação linear [768, 768] que mistura as informações "
                        "das 12 cabeças antes da soma residual. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 768]."
                    ),
                )

                hidden_states = residual_attention + attention_projection
                add_step(
                    name=f"block_{block_index}.residual_after_attention",
                    kind="Residual Add",
                    tensor=hidden_states,
                    block=block_index,
                    description=(
                        "Soma residual: hidden_states = residual_attention + attn_output. "
                        "A entrada original do sub-bloco de atenção é somada à saída processada. "
                        "Isso permite que o gradiente flua diretamente durante o treinamento "
                        "(sem passar pelas camadas internas). "
                        f"Bloco {block_index + 1}/12."
                    ),
                )

                residual_mlp = hidden_states
                add_step(
                    name=f"block_{block_index}.residual_mlp_input",
                    kind="Residual Input",
                    tensor=residual_mlp,
                    block=block_index,
                    description=(
                        "Copia o hidden_states ANTES do MLP (feed-forward). "
                        "Assim como na atenção, será somado de volta após o MLP. "
                        f"Bloco {block_index + 1}/12. "
                        "Cada bloco do transformer tem dois sub-blocos: atenção + MLP, "
                        "cada um com sua própria soma residual."
                    ),
                )

                ln_2_out = block.ln_2(hidden_states)
                add_step(
                    name=f"block_{block_index}.ln_2",
                    kind="LayerNorm",
                    tensor=ln_2_out,
                    block=block_index,
                    description=(
                        "Layer Normalization (ln_2) antes do MLP. "
                        "Mesma operação da ln_1: normaliza para média ~0 e variância ~1, "
                        "depois aplica γ e β aprendidos. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 768]."
                    ),
                )

                mlp_fc = (ln_2_out @ block.mlp.c_fc.weight) + block.mlp.c_fc.bias
                add_step(
                    name=f"block_{block_index}.mlp.c_fc",
                    kind="MLP Expand",
                    tensor=mlp_fc,
                    block=block_index,
                    description=(
                        "Primeira camada do MLP (c_fc = fully connected). "
                        "Expande de 768 → 3072 dimensões (4×). "
                        "Essa expansão permite que o modelo aprenda interações mais complexas "
                        "entre as dimensões. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 3072]."
                    ),
                )

                gelu = F.gelu(mlp_fc)
                add_step(
                    name=f"block_{block_index}.mlp.gelu",
                    kind="GELU",
                    tensor=gelu,
                    block=block_index,
                    description=(
                        "Ativação não-linear GELU (Gaussian Error Linear Unit). "
                        "Similar a ReLU, mas com uma transição suave perto de zero. "
                        "GELU(x) ≈ x · Φ(x) onde Φ é a CDF da normal padrão. "
                        "Introduz não-linearidade — sem ela, o MLP seria só duas lineares. "
                        f"Bloco {block_index + 1}/12."
                    ),
                )

                mlp_projection = (gelu @ block.mlp.c_proj.weight) + block.mlp.c_proj.bias
                add_step(
                    name=f"block_{block_index}.mlp.c_proj",
                    kind="MLP Project",
                    tensor=mlp_projection,
                    block=block_index,
                    description=(
                        "Segunda camada do MLP (c_proj). "
                        "Projeta de volta de 3072 → 768 dimensões. "
                        "Essa é a 'garrafinha' (bottleneck) — expande, aplica não-linearidade, "
                        "depois comprime de volta. "
                        f"Bloco {block_index + 1}/12. Shape: [1, seq_len, 768]."
                    ),
                )

                hidden_states = residual_mlp + mlp_projection
                add_step(
                    name=f"block_{block_index}.residual_after_mlp",
                    kind="Residual Add",
                    tensor=hidden_states,
                    block=block_index,
                    description=(
                        "Soma residual: hidden_states = residual_mlp + mlp_output. "
                        "Segunda soma residual do bloco (depois do MLP). "
                        "A saída final deste bloco vira a entrada do próximo bloco. "
                        f"Bloco {block_index + 1}/12 completo."
                    ),
                )

            final_norm = model.transformer.ln_f(hidden_states)
            add_step(
                name="transformer.ln_f",
                kind="Final LayerNorm",
                tensor=final_norm,
                description=(
                    "Layer Normalization final após todos os 12 blocos. "
                    "Última normalização antes da projeção para o vocabulário. "
                    "Shape: [1, seq_len, 768]."
                ),
            )

            last_token_state = final_norm[:, -1, :]
            add_step(
                name="last_token_state",
                kind="Last Token State",
                tensor=last_token_state,
                description=(
                    "Pega APENAS o estado do ÚLTIMO token (posição -1). "
                    "GPT-2 é um modelo generativo autorregressivo — para prever o PRÓXIMO token, "
                    "só usamos a saída da última posição. Shape: [1, 768]."
                ),
            )

            logits = last_token_state @ model.lm_head.weight.T
            add_step(
                name="lm_head",
                kind="Vocabulary Projection",
                tensor=logits,
                description=(
                    "Projeção linear do estado [1, 768] para o vocabulário [1, 50257]. "
                    "lm_head.weight é uma matriz [50257, 768] — uma linha por palavra no vocabulário. "
                    "O produto escalar com cada linha dá um 'score' (logit) para cada palavra. "
                    "Quanto maior o logit, mais provável a palavra ser a próxima."
                ),
            )

            probabilities = F.softmax(logits, dim=-1)
            add_step(
                name="softmax_vocab",
                kind="Vocabulary Softmax",
                tensor=probabilities,
                description=(
                    "Softmax sobre os 50257 logits transforma scores em probabilidades "
                    "(valores entre 0 e 1, somam 1). "
                    "O token com maior probabilidade é a previsão final do modelo. "
                    "Exibe também o top-10 no painel Output."
                ),
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
