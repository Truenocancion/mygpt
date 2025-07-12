import jax.numpy as jnp
from tqdm import tqdm
from utils import load_encoder_hparams_and_params


# Базовые слои

def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(variance + eps)
    return g * x + b


def linear(x, w, b):
    return x @ w + b


# Блок декодера

def attention(q, k, v, mask):
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


def causal_self_attention(x, c_attn, c_proj):
    x = linear(x, **c_attn)
    q, k, v = jnp.split(x, 3, axis=-1)
    causal_mask = (1 - jnp.tri(x.shape[0])) * -1e10
    x = attention(q, k, v, causal_mask)
    x = linear(x, **c_proj)
    return x


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv = jnp.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=-1), qkv))
    causal_mask = (1 - jnp.tri(x.shape[0])) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = jnp.hstack(out_heads)
    x = linear(x, **c_proj)
    return x


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs: jnp.ndarray, params, n_head, n_tokens_to_generate):
    generated_ids = jnp.zeros((len(inputs) + n_tokens_to_generate,), dtype=int)
    generated_ids = generated_ids.at[:len(inputs)].set(inputs)

    # Генерация заданного количества токенов
    for idx in tqdm(range(n_tokens_to_generate), desc="Generating"):
        logits = gpt(generated_ids[:idx + len(inputs)], **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        generated_ids = generated_ids.at[idx + len(inputs)].set(next_id)

    # Возвращаем только новые токены
    new_tokens = generated_ids[len(inputs):].tolist()
    return new_tokens


def chat(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(jnp.array(input_ids), params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text

