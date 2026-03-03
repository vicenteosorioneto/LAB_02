# Transformer Encoder (Forward Pass) — Laboratório

Implementação **didática** do forward pass de um Encoder Transformer feita do zero com:
- Python 3.x
- `numpy`
- `pandas`

Sem uso de PyTorch, TensorFlow, Keras ou bibliotecas prontas de atenção.

## Objetivo
Implementar apenas o **forward pass** de um Encoder Transformer com **N = 6 camadas**, mantendo clareza matemática e foco em shapes.

Arquivo principal:
- `encoder_transformer_skeleton.py`

---

## Escopo implementado

### 1) Preparação dos dados
- Vocabulário simples com `pandas`
- Conversão de frases para IDs
- Matriz de embeddings com `np.random.randn`
- Construção de `X` com shape `(batch_size, seq_len, d_model)`

### 2) Scaled Dot-Product Attention (manual)
- Inicialização de `W_Q`, `W_K`, `W_V`
- Cálculo de `Q`, `K`, `V` por multiplicação matricial
- `scores = Q @ K^T`
- Escalonamento por `sqrt(d_k)`
- Softmax manual com `np.exp`
- Saída `attn_weights @ V`

### 3) Bloco do Encoder
- Residual connection
- LayerNorm manual (`mean`, `var`, `eps`)
- Feed Forward Network:
  - `W1`, `b1`
  - ReLU
  - `W2`, `b2`

### 4) Empilhamento de 6 camadas
Loop com a ordem:
1. `X_att = SelfAttention(X)`
2. `X_norm1 = LayerNorm(X + X_att)`
3. `X_ffn = FFN(X_norm1)`
4. `X_out = LayerNorm(X_norm1 + X_ffn)`

---

## Shapes esperados
Considere:
- `B = batch_size`
- `L = seq_len`
- `D = d_model`
- `D_k = D_v = D` (neste esqueleto single-head)

### Entrada
- `token_ids`: `(B, L)`
- `E`: `(vocab_size, D)`
- `X`: `(B, L, D)`

### Atenção
- `Q, K, V`: `(B, L, D_k)`
- `scores`: `(B, L, L)`
- `attn_weights`: `(B, L, L)`
- `attn_out`: `(B, L, D_v)`

### Bloco
- `X_norm1`: `(B, L, D)`
- `X_ffn`: `(B, L, D)`
- `X_out`: `(B, L, D)`

### Saída final
- `X_enc` após 6 camadas: `(B, L, D)`

---

## Como executar
No diretório do projeto:

```bash
python encoder_transformer_skeleton.py
```

Se você estiver um nível acima:

```bash
python LAB_02/encoder_transformer_skeleton.py
```

---

## Saída esperada (resumo)
Você verá no terminal os blocos:
- `[PASSO 1]` com shapes de vocabulário, IDs, embeddings e `X`
- `[PASSO 2]` com shapes de `Q`, `K`, `V`, `scores`, `attn_weights`, `attn_out`
- `[PASSO 3]` com shapes de `X_norm1`, `X_ffn`, `X_out_demo`
- `[PASSO 4]` com shape final do encoder e confirmação de 6 mapas de atenção

---

## O que NÃO está incluído (por design)
Para manter o foco no pedido do laboratório, este esqueleto **não** inclui:
- Treinamento/backpropagação
- Otimizador
- Multi-head attention
- Positional encoding
- Máscara de padding/causal

---

## Observações acadêmicas
- A implementação prioriza transparência de cálculo e validação de dimensões.
- É um esqueleto para estudo e extensão manual.
- Você pode evoluir depois para multi-head, máscara e positional encoding mantendo a mesma base de forward.
