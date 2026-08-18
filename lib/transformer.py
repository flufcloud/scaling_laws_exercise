import jax
import numpy as np
from jax import numpy as jnp

from .fused_moe import fused_matmul


DTYPE_ALIASES = {
  'bf16': jnp.bfloat16,
  'bfloat16': jnp.bfloat16,
  'fp16': jnp.float16,
  'float16': jnp.float16,
  'f16': jnp.float16,
  'fp32': jnp.float32,
  'float32': jnp.float32,
  'f32': jnp.float32,
  'fp8': jnp.float8_e4m3fn,
  'fp8_e4m3fn': jnp.float8_e4m3fn,
  'float8_e4m3fn': jnp.float8_e4m3fn,
}


def get_dtype(dtype_name):
  key = str(dtype_name).lower()
  if key not in DTYPE_ALIASES:
    raise ValueError(f"Unsupported dtype {dtype_name}. Use bf16, fp16, fp32, or fp8_e4m3fn.")
  return DTYPE_ALIASES[key]


def get_weight_dtype(config):
  return get_dtype(config.get('weight_dtype', 'bf16'))


def get_compute_dtype(config):
  return get_dtype(config.get('compute_dtype', 'bf16'))


def cast_floating_tree(tree, dtype):
  def cast_leaf(x):
    if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating):
      return x.astype(dtype)
    return x

  return jax.tree.map(cast_leaf, tree)


def to_compute_dtype(x, config):
  return x.astype(get_compute_dtype(config))


def compute_matmul(a, b, config):
  return to_compute_dtype(a, config) @ to_compute_dtype(b, config)


def compute_einsum(spec, a, b, config):
  return jnp.einsum(spec, to_compute_dtype(a, config), to_compute_dtype(b, config))


def compute_fan_in(shape):
  if len(shape) < 2:
    return 1
  return int(np.prod(shape[:-1]))


def numpy_he_normal(rng, shape):
  std = np.sqrt(2.0 / compute_fan_in(shape))
  values = rng.normal(0.0, std, size=shape).astype(np.float32)
  return jnp.asarray(values)


def numpy_normal(rng, shape, std):
  values = rng.normal(0.0, std, size=shape).astype(np.float32)
  return jnp.asarray(values)


def numpy_zeros(shape):
  return jnp.zeros(shape, dtype=jnp.float32)


def init_transformer_block(rng, kernel_init, bias_init, moe_init, config):
  block = {}

  # layer norm params
  gamma = kernel_init((1, config['hidden_dim']))
  beta = bias_init(config['hidden_dim'])

  gamma_linear = kernel_init((1, config['hidden_dim']))
  beta_linear = bias_init(config['hidden_dim'])

  block['layer_norm'] = {'attention': {'gamma': gamma, 'beta': beta},
                         'linear': {'gamma': gamma_linear, 'beta': beta_linear}}

  # attention params
  W_q = kernel_init((config['hidden_dim'], config['num_qheads'], config['attention_dim']))
  W_k = kernel_init((config['hidden_dim'], config['num_kvheads'], config['attention_dim']))
  W_v = kernel_init((config['hidden_dim'], config['num_kvheads'], config['attention_dim']))
  W_o = kernel_init((config['num_qheads'], config['attention_dim'], config['hidden_dim']))

  b_q = bias_init((config['num_qheads'], config['attention_dim']))
  b_k = bias_init((config['num_kvheads'], config['attention_dim']))
  b_v = bias_init((config['num_kvheads'], config['attention_dim']))
  b_o = bias_init(config['hidden_dim'])

  block['attention'] = {'W_q': W_q,
                        'W_k': W_k,
                        'W_v': W_v,
                        'W_o': W_o,
                        'b_q': b_q,
                        'b_k': b_k,
                        'b_v': b_v,
                        'b_o': b_o,}

  # linear params
  def create_linear():
    W_up = kernel_init((config['hidden_dim'], config['mlp_dim']))
    W_down = kernel_init((config['mlp_dim'], config['hidden_dim']))
    b_up = bias_init(config['mlp_dim'])
    b_down = bias_init(config['hidden_dim'])
    return {'W_u': W_up,
            'b_u': b_up,
            'W_d': W_down,
            'b_d': b_down}

  def create_experts():
    W_up = kernel_init((config['num_experts'], config['hidden_dim'], config['mlp_dim']))
    W_down = kernel_init((config['num_experts'], config['mlp_dim'], config['hidden_dim']))
    b_up = bias_init((config['num_experts'], config['mlp_dim']))
    b_down = bias_init((config['num_experts'], config['hidden_dim']))
    return {'W_u': W_up,
            'b_u': b_up,
            'W_d': W_down,
            'b_d': b_down}

  if config['num_experts'] == 1:
    block['linear'] = create_linear()
  else:
    block['linear'] = {'experts': create_experts(),
                       'router': {
                           'kernel': moe_init((config['hidden_dim'], config['num_experts'])),
                           'bias': bias_init(config['num_experts'])
                            }
                       }
  return block

def init_params(config):
  rng = np.random.default_rng(int(config.get('param_seed', 1010101)))
  bias_init = numpy_zeros
  moe_init = lambda shape: numpy_normal(rng, shape, 0.0001)
  kernel_init = lambda shape: numpy_he_normal(rng, shape)
  params = {}

  # init embedding and unembedding matrices
  token_embedding = kernel_init((config['vocab_size'], config['hidden_dim']))
  token_bias = bias_init(config['hidden_dim'])

  token_unembedding = kernel_init((config['hidden_dim'], config['vocab_size']))
  token_bias_unembed = bias_init((config['vocab_size']))

  params['embedding'] = {'kernel': token_embedding, 'bias': token_bias}
  params['unembedding'] = {'kernel': token_unembedding, 'bias': token_bias_unembed}

  # init transformer blocks
  params['blocks'] = [init_transformer_block(rng, kernel_init, bias_init, moe_init, config) for _ in range(config['num_layers'])]

  return cast_floating_tree(params, get_weight_dtype(config))

def apply_embedding(batch, train_state, config):
  batch = jax.nn.one_hot(batch, config['vocab_size'], axis=-1)
  batch = compute_matmul(batch, train_state['kernel'], config) + train_state['bias'].astype(get_compute_dtype(config))

  # calculate the positional embedding
  seq_ind = jnp.arange(batch.shape[1]).reshape(batch.shape[1], 1)
  evens = (jnp.arange(config['hidden_dim']) % 2)
  odds = (jnp.arange(1, config['hidden_dim'] + 1) % 2)
  pos_embed_denom = 1000 ** (((jnp.arange(config['hidden_dim']) // 2) * 2) / config['hidden_dim'])
  pos_embed = seq_ind / pos_embed_denom
  sins = jnp.sin(pos_embed)
  cosn = jnp.cos(pos_embed)
  pos_embed = (sins * odds) + (cosn * evens)
  return batch + pos_embed.astype(batch.dtype)

def apply_layer_norm(batch, train_state, config):
  normed = jax.nn.standardize(batch.astype(jnp.float32), epsilon=config['norm_eps'])
  normed = (normed * train_state['gamma'].astype(jnp.float32)) + train_state['beta'].astype(jnp.float32)
  return normed.astype(get_compute_dtype(config))

def get_linear_load_balancing_ratio(batch, params, config):
  if config['num_experts'] == 1:
    return None

  b, s = batch.shape[:2]
  k, e = config['top_k'], config['num_experts']
  router_logits = compute_matmul(batch, params['router']['kernel'], config) + params['router']['bias']
  all_router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1).astype(get_compute_dtype(config))
  _, routes = jax.lax.top_k(all_router_probs, k)
  flattened_routes = routes.flatten()
  g = jnp.bincount(flattened_routes, length=e)
  return jnp.max(g / (b * s * k))


def apply_linear(batch, params, load_balancing_loss, config):
  if config['num_experts'] == 1:
    up_proj = compute_matmul(batch, params['W_u'], config) + params['b_u']
    activations = jax.nn.gelu(up_proj)
    down_proj = compute_matmul(activations, params['W_d'], config) + params['b_d']
    return down_proj, load_balancing_loss
  b, s, d = batch.shape
  k, e = config['top_k'], config['num_experts']
  alpha, num_layers = config['load_balancing_alpha'], config['num_layers']
  router_logits = compute_matmul(batch, params['router']['kernel'], config) + params['router']['bias']
  all_router_probs = jax.nn.softmax(router_logits.astype(jnp.float32), axis=-1).astype(get_compute_dtype(config))
  router_probs, routes = jax.lax.top_k(all_router_probs, k)

  flattened_routes = routes.flatten()
  pack_routes = jnp.argsort(flattened_routes, axis=0)
  seq_indices = jnp.argsort(pack_routes)
  g = jnp.bincount(flattened_routes, length=e)

  fully_batched_activations = batch.reshape(b * s, d)
  fully_batched_activations = jnp.repeat(fully_batched_activations, k, axis=0)
  packed_activations = fully_batched_activations[pack_routes]

  fully_batched_router_probs = router_probs.reshape(b * s * k, 1)
  packed_router_probs = fully_batched_router_probs[pack_routes]

  if not config['use_custom_kernel']:
    bias_routes = jnp.sort(flattened_routes, axis=0)
    proj = jax.lax.ragged_dot(packed_activations, params['experts']['W_u'], g)
    proj = proj + params['experts']['b_u'][bias_routes]
    proj = jax.nn.gelu(proj)
    proj = jax.lax.ragged_dot(proj, params['experts']['W_d'], g)
    proj = proj + params['experts']['b_d'][bias_routes]
  else:
    proj = fused_matmul(packed_activations,
                        params['experts']['W_u'],
                        params['experts']['W_d'],
                        params['experts']['b_u'],
                        params['experts']['b_d'],
                        g,
                        BLOCK_B=config['block_sizes']['b'],
                        BLOCK_D=config['block_sizes']['d'],
                        BLOCK_F=config['block_sizes']['f'])

  # allows the gradient to flow through to the router and so on
  proj = proj * packed_router_probs
  proj = proj[seq_indices]
  proj = proj.reshape(b, s, k, d)
  proj = proj.sum(axis=2)

  router_fract = g / (b * s * k)
  router_fract_prob = jnp.sum(all_router_probs, axis=(0, 1)) / (b * s * k)
  load_balancing_loss += ((1 / num_layers) * alpha * e) * jnp.dot(router_fract, router_fract_prob)
  return proj, load_balancing_loss

def apply_attention(batch, params, config):
  W_k = params['W_k']
  W_q = params['W_q']
  W_v = params['W_v']
  W_o = params['W_o']
  b_k = params['b_k']
  b_q = params['b_q']
  b_v = params['b_v']
  b_o = params['b_o']

  B, T, _ = batch.shape

  num_q_heads = config['num_qheads']
  num_kv_heads = config['num_kvheads']
  g = num_q_heads // num_kv_heads
  head_dim = config['attention_dim']

  X_k = compute_einsum('btd,dkh->btkh', batch, W_k, config) + b_k
  X_q = compute_einsum('btd,dnh->btnh', batch, W_q, config) + b_q
  X_v = compute_einsum('btd,dkh->btkh', batch, W_v, config) + b_v

  # [B, T, num_heads, H] -> [B, T, num_kv_heads, g, H]
  X_q = X_q.reshape(B, T, num_kv_heads, g, head_dim)

  # scores: [B, T, S, num_kv_heads, g]
  scores = compute_einsum('btkgh,bskh->btskg', X_q, X_k, config)

  # masks (preserves only the lower triangle, so the upper triangle ie lookahead equals 0)
  score_mask = jnp.tri(T)[None, :, :, None, None]
  scores = jnp.where(score_mask, scores, jnp.finfo(scores.dtype).min)
  scores = scores / jnp.sqrt(head_dim).astype(scores.dtype)

  avgs = jax.nn.softmax(scores.astype(jnp.float32), axis=2).astype(get_compute_dtype(config))

  # attended: [B, T, num_kv_heads, g, H]
  attended = compute_einsum('btskg,bskh->btkgh', avgs, X_v, config)

  # [B, T, num_heads, H]
  attended = attended.reshape(B, T, num_q_heads, head_dim)

  # output projection
  return compute_einsum('btnh,nhd->btd', attended, W_o, config) + b_o

def apply_block(batch, params, load_balancing_loss, config):
  attended = apply_attention(batch, params['attention'], config)
  batch = batch + apply_layer_norm(attended, params['layer_norm']['attention'], config)
  activations, load_balancing_loss = apply_linear(batch, params['linear'], load_balancing_loss, config)
  batch = batch + apply_layer_norm(activations, params['layer_norm']['linear'], config)
  return batch, load_balancing_loss

def apply_model(params, batch, config):
  load_balancing_loss = 0
  batch = apply_embedding(batch, params['embedding'], config)
  for i in range(config['num_layers']):
    batch, load_balancing_loss = apply_block(batch, params['blocks'][i], load_balancing_loss, config)

  logits = compute_matmul(batch, params['unembedding']['kernel'], config) + params['unembedding']['bias'].astype(get_compute_dtype(config))
  return logits, load_balancing_loss

def get_load_balancing_ratio(params, batch, config):
  if config['num_experts'] == 1:
    return None

  ratios = []
  batch = apply_embedding(batch, params['embedding'], config)
  for i in range(config['num_layers']):
    block_params = params['blocks'][i]
    attended = apply_attention(batch, block_params['attention'], config)
    batch = batch + apply_layer_norm(attended, block_params['layer_norm']['attention'], config)
    ratios.append(get_linear_load_balancing_ratio(batch, block_params['linear'], config))
    activations, _ = apply_linear(batch, block_params['linear'], 0, config)
    batch = batch + apply_layer_norm(activations, block_params['layer_norm']['linear'], config)

  return jnp.max(jnp.array(ratios))

