import numpy as np
import pandas as pd

"""
LAB: Encoder Transformer (forward pass only) - ESQUELETO GUIADO
Restrições atendidas:
- Python 3.x
- Apenas numpy e pandas
- Sem PyTorch/TensorFlow/Keras
- Sem implementação "pronta" de biblioteca de atenção

Objetivo deste arquivo:
- Mostrar passo a passo, na ordem correta, como montar o forward pass
- Explicar dimensões (shapes) em cada etapa
- Manter estrutura didática para você completar/manualizar
"""

# ==========================================================
# PASSO 0) HIPERPARÂMETROS (ajuste conforme o laboratório)
# ==========================================================
BATCH_SIZE = 2
SEQ_LEN = 6
D_MODEL = 16
D_FF = 64
N_LAYERS = 6

# Para simplificar o laboratório: single-head com d_k = d_v = d_model
D_K = D_MODEL
D_V = D_MODEL


# ==========================================================
# PASSO 1) PREPARAÇÃO DOS DADOS
# - vocabulário com pandas
# - frase -> IDs
# - embedding matrix com np.random.randn
# - tensor X com shape (batch_size, seq_len, d_model)
# ==========================================================

# 1.1 Exemplo de frases base (troque pelo seu dataset)
corpus = [
    "eu gosto de nlp",
    "transformers usam atencao",
    "eu estudo modelos",
]

# 1.2 Tokenização simples com pandas
#     tokens_series[i] será lista de tokens da frase i
tokens_series = pd.Series(corpus).str.lower().str.split()

# 1.3 Construção de vocabulário simples
all_tokens = tokens_series.explode().dropna()
unique_tokens = pd.Series(all_tokens.unique(), name="token")

# Inclui tokens especiais (PAD e UNK)
vocab_df = pd.concat(
    [pd.DataFrame({"token": ["<PAD>", "<UNK>"]}), unique_tokens.to_frame()],
    ignore_index=True,
)
vocab_df = vocab_df.drop_duplicates(subset="token").reset_index(drop=True)
vocab_df["id"] = np.arange(len(vocab_df), dtype=np.int64)

# Dicionários de mapeamento
TOKEN_TO_ID = dict(zip(vocab_df["token"], vocab_df["id"]))
ID_TO_TOKEN = dict(zip(vocab_df["id"], vocab_df["token"]))

VOCAB_SIZE = len(vocab_df)


def sentence_to_ids(sentence, token_to_id, seq_len):
    """
    Converte uma frase em IDs com padding/truncamento.

    Entrada:
      sentence: str
      token_to_id: dict
      seq_len: int

    Saída:
      ids: lista com comprimento seq_len
    """
    tokens = sentence.lower().split()
    ids = [token_to_id.get(tok, token_to_id["<UNK>"]) for tok in tokens]

    if len(ids) < seq_len:
        ids = ids + [token_to_id["<PAD>"]] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]

    return ids


# 1.4 Mini-batch de exemplo
batch_sentences = [
    "eu gosto de nlp",
    "eu estudo atencao",
]

# token_ids -> shape (BATCH_SIZE, SEQ_LEN)
token_ids = np.array(
    [sentence_to_ids(s, TOKEN_TO_ID, SEQ_LEN) for s in batch_sentences],
    dtype=np.int64,
)

# 1.5 Embeddings
# E -> shape (VOCAB_SIZE, D_MODEL)
E = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.02

# 1.6 Lookup para montar X
# X -> shape (BATCH_SIZE, SEQ_LEN, D_MODEL)
X = E[token_ids]

print("[PASSO 1]")
print("vocab_df shape:", vocab_df.shape)         # (VOCAB_SIZE, 2)
print("token_ids shape:", token_ids.shape)       # (BATCH_SIZE, SEQ_LEN)
print("E shape:", E.shape)                       # (VOCAB_SIZE, D_MODEL)
print("X shape:", X.shape)                       # (BATCH_SIZE, SEQ_LEN, D_MODEL)
print("-" * 50)


# ==========================================================
# PASSO 2) SCALED DOT-PRODUCT ATTENTION (manual)
# - W_Q, W_K, W_V
# - Q, K, V
# - scores = Q @ K^T
# - scaling por sqrt(d_k)
# - softmax manual com np.exp
# - saída = atenção @ V
# ==========================================================

def softmax_manual(x, axis=-1):
    """
    Softmax manual com estabilidade numérica.

    x: array qualquer
    axis: dimensão onde aplicar softmax
    """
    # Subtrair max evita overflow numérico em exp
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def init_attention_params(d_model, d_k, d_v):
    """
    Inicializa W_Q, W_K, W_V.

    Shapes esperados:
      W_Q: (d_model, d_k)
      W_K: (d_model, d_k)
      W_V: (d_model, d_v)
    """
    params = {
        "W_Q": np.random.randn(d_model, d_k) * 0.02,
        "W_K": np.random.randn(d_model, d_k) * 0.02,
        "W_V": np.random.randn(d_model, d_v) * 0.02,
    }
    return params


def scaled_dot_product_attention(X_in, W_Q, W_K, W_V):
    """
    Atenção self-attention single-head (forward only).

    Entrada:
      X_in: (B, L, D)
      W_Q:  (D, d_k)
      W_K:  (D, d_k)
      W_V:  (D, d_v)

    Saída:
      attn_out: (B, L, d_v)
      attn_weights: (B, L, L)
      aux: dict com intermediários (didático)
    """
    # Q, K, V
    Q = X_in @ W_Q                   # (B, L, d_k)
    K = X_in @ W_K                   # (B, L, d_k)
    V = X_in @ W_V                   # (B, L, d_v)

    # scores = Q @ K^T
    K_t = np.transpose(K, (0, 2, 1)) # (B, d_k, L)
    scores = Q @ K_t                 # (B, L, L)

    # scaling
    d_k = W_Q.shape[1]
    scaled_scores = scores / np.sqrt(d_k)  # (B, L, L)

    # softmax manual
    attn_weights = softmax_manual(scaled_scores, axis=-1)  # (B, L, L)

    # output
    attn_out = attn_weights @ V      # (B, L, d_v)

    aux = {
        "Q": Q,
        "K": K,
        "V": V,
        "scores": scores,
        "scaled_scores": scaled_scores,
    }
    return attn_out, attn_weights, aux


att_params_demo = init_attention_params(D_MODEL, D_K, D_V)
X_att_demo, att_w_demo, att_aux_demo = scaled_dot_product_attention(
    X,
    att_params_demo["W_Q"],
    att_params_demo["W_K"],
    att_params_demo["W_V"],
)

print("[PASSO 2]")
print("Q shape:", att_aux_demo["Q"].shape)                  # (B, L, D_K)
print("K shape:", att_aux_demo["K"].shape)                  # (B, L, D_K)
print("V shape:", att_aux_demo["V"].shape)                  # (B, L, D_V)
print("scores shape:", att_aux_demo["scores"].shape)        # (B, L, L)
print("attn_weights shape:", att_w_demo.shape)               # (B, L, L)
print("attn_out shape:", X_att_demo.shape)                   # (B, L, D_V)
print("-" * 50)


# ==========================================================
# PASSO 3) RESIDUAL + LAYERNORM + FFN
# - Residual connection
# - LayerNorm manual (mean, var, epsilon)
# - FFN com W1,b1 -> ReLU -> W2,b2
# ==========================================================

def layer_norm_manual(X_in, gamma, beta, eps=1e-5):
    """
    LayerNorm na última dimensão (D).

    Entrada:
      X_in:  (B, L, D)
      gamma: (D,)
      beta:  (D,)

    Saída:
      Y: (B, L, D)
    """
    mean = np.mean(X_in, axis=-1, keepdims=True)              # (B, L, 1)
    var = np.mean((X_in - mean) ** 2, axis=-1, keepdims=True) # (B, L, 1)
    X_hat = (X_in - mean) / np.sqrt(var + eps)                # (B, L, D)
    Y = gamma * X_hat + beta                                  # (B, L, D)
    return Y


def relu(x):
    return np.maximum(0, x)


def init_ffn_params(d_model, d_ff):
    """
    Inicializa parâmetros da FFN.

    Shapes:
      W1: (d_model, d_ff)
      b1: (d_ff,)
      W2: (d_ff, d_model)
      b2: (d_model,)
    """
    return {
        "W1": np.random.randn(d_model, d_ff) * 0.02,
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model) * 0.02,
        "b2": np.zeros(d_model),
    }


def ffn_forward(X_in, W1, b1, W2, b2):
    """
    FFN token-wise.

    Entrada:
      X_in: (B, L, D)

    Saída:
      out: (B, L, D)
    """
    h = X_in @ W1 + b1   # (B, L, D_FF)
    h = relu(h)          # (B, L, D_FF)
    out = h @ W2 + b2    # (B, L, D)
    return out


# Demo de um bloco (sem empilhar ainda)
ln_gamma1 = np.ones(D_MODEL)
ln_beta1 = np.zeros(D_MODEL)
ln_gamma2 = np.ones(D_MODEL)
ln_beta2 = np.zeros(D_MODEL)

ffn_params_demo = init_ffn_params(D_MODEL, D_FF)

# Atenção
X_att = X_att_demo                                  # (B, L, D)

# Residual + LN 1
X_norm1 = layer_norm_manual(X + X_att, ln_gamma1, ln_beta1)  # (B, L, D)

# FFN
X_ffn = ffn_forward(
    X_norm1,
    ffn_params_demo["W1"],
    ffn_params_demo["b1"],
    ffn_params_demo["W2"],
    ffn_params_demo["b2"],
)                                                   # (B, L, D)

# Residual + LN 2
X_out_demo = layer_norm_manual(X_norm1 + X_ffn, ln_gamma2, ln_beta2)  # (B, L, D)

print("[PASSO 3]")
print("X_norm1 shape:", X_norm1.shape)            # (B, L, D)
print("X_ffn shape:", X_ffn.shape)                # (B, L, D)
print("X_out_demo shape:", X_out_demo.shape)      # (B, L, D)
print("-" * 50)


# ==========================================================
# PASSO 4) EMPILHAR N=6 CAMADAS (loop)
# Ordem exigida:
#   X_att = SelfAttention(X)
#   X_norm1 = LayerNorm(X + X_att)
#   X_ffn = FFN(X_norm1)
#   X_out = LayerNorm(X_norm1 + X_ffn)
# ==========================================================

def init_encoder_layer_params(d_model, d_k, d_v, d_ff):
    """
    Inicializa parâmetros de UMA camada do encoder.
    """
    att = init_attention_params(d_model, d_k, d_v)
    ffn = init_ffn_params(d_model, d_ff)

    layer = {
        # Attention
        "W_Q": att["W_Q"],
        "W_K": att["W_K"],
        "W_V": att["W_V"],

        # LN1
        "gamma1": np.ones(d_model),
        "beta1": np.zeros(d_model),

        # FFN
        "W1": ffn["W1"],
        "b1": ffn["b1"],
        "W2": ffn["W2"],
        "b2": ffn["b2"],

        # LN2
        "gamma2": np.ones(d_model),
        "beta2": np.zeros(d_model),
    }
    return layer


# 4.1 Inicializa lista de 6 camadas
encoder_layers = [
    init_encoder_layer_params(D_MODEL, D_K, D_V, D_FF)
    for _ in range(N_LAYERS)
]

# 4.2 Forward no stack
X_enc = X.copy()   # (B, L, D)

all_attention_weights = []

for layer_idx in range(N_LAYERS):
    p = encoder_layers[layer_idx]

    # (a) Self-Attention
    X_att, att_w, _ = scaled_dot_product_attention(
        X_enc,
        p["W_Q"],
        p["W_K"],
        p["W_V"],
    )

    # (b) Residual + LayerNorm
    X_norm1 = layer_norm_manual(
        X_enc + X_att,
        p["gamma1"],
        p["beta1"],
    )

    # (c) FFN
    X_ffn = ffn_forward(
        X_norm1,
        p["W1"],
        p["b1"],
        p["W2"],
        p["b2"],
    )

    # (d) Residual + LayerNorm
    X_out = layer_norm_manual(
        X_norm1 + X_ffn,
        p["gamma2"],
        p["beta2"],
    )

    # saída da camada atual vira entrada da próxima
    X_enc = X_out
    all_attention_weights.append(att_w)

print("[PASSO 4]")
print("X final do encoder shape:", X_enc.shape)  # (B, L, D)
print("num mapas de atenção:", len(all_attention_weights))  # 6
print("shape de cada mapa de atenção:", all_attention_weights[0].shape)  # (B, L, L)
print("-" * 50)


# ==========================================================
# PASSO 5) CHECKLIST DE CONFERÊNCIA (para seu relatório)
# ==========================================================

# 1) Dados:
#    token_ids -> (B, L)
#    E -> (V, D)
#    X -> (B, L, D)
#
# 2) Atenção:
#    Q,K,V -> (B, L, D_k)
#    scores -> (B, L, L)
#    attn_weights -> (B, L, L)
#    attn_out -> (B, L, D_v)
#
# 3) Bloco:
#    X_norm1 -> (B, L, D)
#    X_ffn -> (B, L, D)
#    X_out -> (B, L, D)
#
# 4) Stack N=6:
#    saída final X_enc -> (B, L, D)


if __name__ == "__main__":
    # O script já executa o fluxo acima ao rodar.
    # Este bloco existe apenas para marcar execução intencional.
    pass
