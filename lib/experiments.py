import copy
import hashlib
import secrets
from functools import lru_cache

from .data import create_all_addition_token_dataset, create_training_token_dataset
from .evaluation import eval_model
from .training import init_train_state, train


PARAM_SUFFIXES = {
  'k': 1e3,
  'm': 1e6,
  'b': 1e9,
}
PARAM_LOOKUP_MIN = 100000
PARAM_LOOKUP_MAX = 500000000
LEGACY_PARAM_LOOKUP_MIN = 10000
LEGACY_PARAM_LOOKUP_MAX = 40000000
PARAM_LOOKUP_INTERVAL = 10000


def derive_rng_seed(run_seed, stream_name):
  payload = f'{int(run_seed)}:{stream_name}'.encode('utf-8')
  return int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')


def prepare_run_rng(config):
  if config.get('run_seed') is None and config.get('param_seed') is not None:
    config.setdefault('data_seed', 12)
    config.setdefault('batch_sampling_seed', config['data_seed'])
    config['rng_scheme'] = 'explicit_parameter_and_data_seeds'
    return config

  if config.get('run_seed') is None:
    config['run_seed'] = secrets.randbits(128)
  run_seed = int(config['run_seed'])
  config['param_seed'] = derive_rng_seed(run_seed, 'parameters')
  config['data_seed'] = derive_rng_seed(run_seed, 'batch_sampling')
  config['batch_sampling_seed'] = config['data_seed']
  config['rng_scheme'] = 'sha256_streams_from_128bit_run_seed_v1'
  config['training_dataset'] = 'all_ordered_additions_0_to_999_sampled_with_replacement'
  config['training_dataset_size'] = 1_000_000
  return config


def parse_num_params_value(value):
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return int(round(value))

  text = str(value).strip().lower().replace('_', '').replace(',', '')
  if not text:
    raise ValueError('parameter count cannot be empty')
  multiplier = 1
  if text[-1] in PARAM_SUFFIXES:
    multiplier = PARAM_SUFFIXES[text[-1]]
    text = text[:-1]
  return int(round(float(text) * multiplier))


def supported_num_param_values(is_moe=False, legacy_configs=False):
  lower = LEGACY_PARAM_LOOKUP_MIN if legacy_configs else PARAM_LOOKUP_MIN
  upper = LEGACY_PARAM_LOOKUP_MAX if legacy_configs else PARAM_LOOKUP_MAX
  values = []
  ratio = 1

  while True:
    config = get_config_for_model_size(ratio, is_moe=is_moe)
    num_params = estimate_params_and_flops(config)[0]
    if num_params > upper:
      break
    if num_params >= lower:
      values.append(int(num_params))
    ratio += 1

  return values


def supported_num_param_targets(is_moe=False, legacy_configs=False):
  return supported_num_param_values(is_moe=is_moe, legacy_configs=legacy_configs)


def nearest_lookup_num_params(target_num_params, is_moe=False, legacy_configs=False, clamp=False):
  target_num_params = parse_num_params_value(target_num_params)
  lower = LEGACY_PARAM_LOOKUP_MIN if legacy_configs else PARAM_LOOKUP_MIN
  upper = LEGACY_PARAM_LOOKUP_MAX if legacy_configs else PARAM_LOOKUP_MAX
  if target_num_params < lower or target_num_params > upper:
    if not clamp:
      raise ValueError(
        f'target_num_params must be between {lower} and {upper}'
      )

  return min(max(target_num_params, lower), upper)


def run_experiment(config, jit, train_set=None):
  num_params, activated_params, flops, _ = estimate_params_and_flops(config)
  config['num_params'] = num_params
  config['activated_params'] = activated_params
  config['flops'] = flops
  config.setdefault('nominal_flops', flops)

  prepare_run_rng(config)

  if train_set is None:
    train_set = create_all_addition_token_dataset()

  train_state = init_train_state(config)
  trained_params, train_loss = train(train_state, train_set, config, jit)

  if config.get('val_num_tokens') is not None:
    val_set = create_training_token_dataset(
      int(config['val_num_tokens']),
      seed=int(config.get('val_seed', 13)),
    )
    val_loss, digit_error_rows = eval_model(
      trained_params,
      val_set,
      None,
      config,
      return_digit_errors=True,
    )
    config['val_loss'] = float(val_loss)
    config['val_examples_evaluated'] = sum(row['num_examples'] for row in digit_error_rows)
    config['val_exact_answer_correct'] = sum(row['num_correct'] for row in digit_error_rows)
    if config['val_examples_evaluated']:
      config['val_exact_answer_accuracy_percent'] = (
        100 * config['val_exact_answer_correct'] / config['val_examples_evaluated']
      )
      config['val_exact_answer_error_percent'] = 100 - config['val_exact_answer_accuracy_percent']
    config['val_accuracy_definition'] = 'teacher_forced_exact_answer_digits'

  return trained_params, train_loss


def run_experiment_batch(configs, jit):
  if len(configs) == 0:
    return []

  shared_train_set = create_all_addition_token_dataset()
  results = []

  for config in configs:
    _, train_loss = run_experiment(
      config,
      jit=jit,
      train_set=shared_train_set,
    )
    val_loss = config.get('val_loss')
    results.append({
      'train_loss': None if train_loss is None else float(train_loss),
      'val_loss': None if val_loss is None else float(val_loss),
      'val_exact_answer_accuracy_percent': config.get('val_exact_answer_accuracy_percent'),
      'val_exact_answer_error_percent': config.get('val_exact_answer_error_percent'),
    })

  return results


def estimate_params_and_flops(config):
  # upper line is weights, lower line is biases
  attention_params = config['num_layers'] * 2 * config['hidden_dim'] * (config['num_qheads'] + config['num_kvheads']) * config['attention_dim']
  attention_params += config['num_layers'] * ((((2 * config['num_qheads']) + config['num_kvheads']) * config['attention_dim']) + config['hidden_dim'])

  linear_params = 2 * config['num_layers']  * config['hidden_dim'] * config['mlp_dim']
  linear_params += config['num_layers'] * (config['hidden_dim'] + config['mlp_dim'])

  layer_norm_params = 4 * config['num_layers'] * config['hidden_dim']
  embed_params = (2 * config['vocab_size'] * config['hidden_dim']) + config['vocab_size'] +  config['hidden_dim']

  num_params = attention_params + (linear_params * config['num_experts'])  + layer_norm_params + embed_params
  if config['num_experts'] == 1:
    activated_params = num_params
  else:
    activated_params = attention_params + (linear_params * config['top_k']) + layer_norm_params + embed_params

  flops = 6 * activated_params * config['num_tokens']
  return num_params, activated_params, flops, (linear_params, attention_params, layer_norm_params + embed_params)

def get_config_for_model_size(
  ratio,
  is_moe=False,
  use_kernel=False,
  *,
  init_lr=0.0,
  max_lr=1.0e-5,
  decay_alpha=0.1,
  warmup_fraction=0.05,
  adam_beta1=0.9,
  adam_beta2=0.99,
  adam_eps=1e-8,
  grad_clip_norm=1.0,
  load_balancing_alpha=0.01,
  num_experts=4,
  top_k=2,
  batch_size=32,
  val_num_tokens=50000,
  val_seed=13,
  weight_dtype='bf16',
  compute_dtype='bf16',
  BLOCK_B=256,
  BLOCK_D=1024,
  BLOCK_F=4096,
):
  ratio = int(ratio)
  if ratio < 1:
    raise ValueError('ratio must be >= 1')
  if init_lr < 0 or max_lr <= 0 or init_lr > max_lr:
    raise ValueError('learning rates must satisfy 0 <= init_lr <= max_lr and max_lr > 0')
  if decay_alpha < 0 or decay_alpha > 1:
    raise ValueError('decay_alpha must be between 0 and 1')
  if warmup_fraction < 0 or warmup_fraction >= 1:
    raise ValueError('warmup_fraction must be between 0 (inclusive) and 1 (exclusive)')

  attention_dim = 16
  hidden_dim = 64 + (16 * (ratio - 1))
  config = {
    'num_tokens': 100000,
    'epochs': 1,
    'batch_size': int(batch_size),
    'init_lr': init_lr,
    'max_lr': max_lr,
    'decay_alpha': decay_alpha,
    'warmup_fraction': warmup_fraction,
    'adam_beta1': adam_beta1,
    'adam_beta2': adam_beta2,
    'adam_eps': adam_eps,
    'grad_clip_norm': grad_clip_norm,
    'norm_eps': 1e-5,
    'mini_test_set': None,
    'vocab_size': 13,
    'hidden_dim': hidden_dim,
    'mlp_dim': hidden_dim * 4,
    'activation': 'gelu',
    'attention_dim': attention_dim,
    'num_qheads': hidden_dim // attention_dim,
    'num_kvheads': hidden_dim // attention_dim,
    'num_layers': 2,
    'num_experts': 1,
    'top_k': 1,
    'load_balancing_alpha': load_balancing_alpha,
    'use_custom_kernel': use_kernel,
    'weight_dtype': weight_dtype,
    'compute_dtype': compute_dtype,
    'block_sizes': {
        'b': BLOCK_B,
        'd': BLOCK_D,
        'f': BLOCK_F,
    },
    'param_seed': 1010101,
    'data_seed': 12,
    'val_seed': int(val_seed),
    'val_num_tokens': int(val_num_tokens),
    'ratio': ratio,
  }

  if is_moe:
    config['num_experts'] = int(num_experts)
    config['top_k'] = int(top_k)
    if config['num_experts'] < 2:
      raise ValueError('num_experts must be >= 2 for MoE configs')
    if config['top_k'] < 1 or config['top_k'] > config['num_experts']:
      raise ValueError('top_k must be between 1 and num_experts')

  return config


@lru_cache(maxsize=4096)
def _get_config_for_num_params_cached(
  target_num_params,
  is_moe=False,
  use_kernel=False,
  *,
  init_lr=0.0,
  max_lr=1.0e-5,
  decay_alpha=0.1,
  warmup_fraction=0.05,
  adam_beta1=0.9,
  adam_beta2=0.99,
  adam_eps=1e-8,
  grad_clip_norm=1.0,
  load_balancing_alpha=0.01,
  num_experts=4,
  top_k=2,
  batch_size=32,
  val_num_tokens=50000,
  val_seed=13,
  weight_dtype='bf16',
  compute_dtype='bf16',
  BLOCK_B=256,
  BLOCK_D=1024,
  BLOCK_F=4096,
  min_hidden_dim=32,
  max_hidden_dim=4096,
  hidden_multiple=16,
  min_mlp_ratio=2.0,
  max_mlp_ratio=5.0,
):
  target_num_params = parse_num_params_value(target_num_params)
  if target_num_params < LEGACY_PARAM_LOOKUP_MIN or target_num_params > PARAM_LOOKUP_MAX:
    raise ValueError(
      f'target_num_params must be between {LEGACY_PARAM_LOOKUP_MIN} and {PARAM_LOOKUP_MAX}'
    )

  def candidate_config(hidden_dim, mlp_dim):
    config = get_config_for_model_size(
      1,
      is_moe=is_moe,
      use_kernel=use_kernel,
      init_lr=init_lr,
      max_lr=max_lr,
      decay_alpha=decay_alpha,
      warmup_fraction=warmup_fraction,
      adam_beta1=adam_beta1,
      adam_beta2=adam_beta2,
      adam_eps=adam_eps,
      grad_clip_norm=grad_clip_norm,
      load_balancing_alpha=load_balancing_alpha,
      num_experts=num_experts,
      top_k=top_k,
      batch_size=batch_size,
      val_num_tokens=val_num_tokens,
      val_seed=val_seed,
      weight_dtype=weight_dtype,
      compute_dtype=compute_dtype,
      BLOCK_B=BLOCK_B,
      BLOCK_D=BLOCK_D,
      BLOCK_F=BLOCK_F,
    )
    config['hidden_dim'] = int(hidden_dim)
    config['mlp_dim'] = int(mlp_dim)
    config['attention_dim'] = 16
    config['num_qheads'] = max(1, int(hidden_dim) // config['attention_dim'])
    config['num_kvheads'] = max(1, int(hidden_dim) // config['attention_dim'])
    config['num_layers'] = 2
    config.pop('ratio', None)
    config['requested_num_params'] = target_num_params
    config['param_target_kind'] = 'num_params'
    return config

  best = None
  hidden_start = max(hidden_multiple, ((int(min_hidden_dim) + hidden_multiple - 1) // hidden_multiple) * hidden_multiple)
  hidden_stop = ((int(max_hidden_dim) // hidden_multiple) * hidden_multiple) + 1

  for hidden_dim in range(hidden_start, hidden_stop, int(hidden_multiple)):
    min_mlp_dim = max(1, int(round(min_mlp_ratio * hidden_dim)))
    max_mlp_dim = max(min_mlp_dim, int(round(max_mlp_ratio * hidden_dim)))
    base_config = candidate_config(hidden_dim, 1)
    base_active = estimate_params_and_flops(base_config)[0]
    next_config = candidate_config(hidden_dim, 2)
    next_active = estimate_params_and_flops(next_config)[0]
    mlp_coeff = next_active - base_active
    if mlp_coeff <= 0:
      continue

    mlp_guess = int(round(1 + ((target_num_params - base_active) / mlp_coeff)))
    mlp_candidates = {min_mlp_dim, max_mlp_dim}
    for mlp_dim in range(mlp_guess - 32, mlp_guess + 33):
      if min_mlp_dim <= mlp_dim <= max_mlp_dim:
        mlp_candidates.add(int(mlp_dim))

    for mlp_dim in mlp_candidates:
      config = candidate_config(hidden_dim, mlp_dim)
      num_params, activated_params, _, _ = estimate_params_and_flops(config)
      param_error = abs(num_params - target_num_params)
      mlp_ratio_error = abs((mlp_dim / hidden_dim) - 4.0)
      score = (param_error, mlp_ratio_error, hidden_dim)
      if best is None or score < best[0]:
        best = (score, config, num_params, activated_params)

  if best is None:
    raise ValueError(f'Could not find a valid config near {target_num_params} parameters')

  _, config, num_params, activated_params = best
  config['num_params'] = int(num_params)
  config['activated_params'] = int(activated_params)
  config['param_error'] = int(num_params - target_num_params)
  config['param_relative_error'] = float(config['param_error'] / target_num_params)
  config['mlp_ratio'] = float(config['mlp_dim'] / config['hidden_dim'])
  return config


def get_config_for_num_params(
  target_num_params,
  is_moe=False,
  use_kernel=False,
  *,
  legacy_configs=False,
  clamp=False,
  init_lr=0.0,
  max_lr=1.0e-5,
  decay_alpha=0.1,
  warmup_fraction=0.05,
  adam_beta1=0.9,
  adam_beta2=0.99,
  adam_eps=1e-8,
  grad_clip_norm=1.0,
  load_balancing_alpha=0.01,
  num_experts=4,
  top_k=2,
  batch_size=32,
  val_num_tokens=50000,
  val_seed=13,
  weight_dtype='bf16',
  compute_dtype='bf16',
  BLOCK_B=256,
  BLOCK_D=1024,
  BLOCK_F=4096,
):
  requested_num_params = parse_num_params_value(target_num_params)
  lookup_num_params = nearest_lookup_num_params(
    requested_num_params, is_moe=is_moe, legacy_configs=legacy_configs, clamp=clamp,
  )

  config = copy.deepcopy(_get_config_for_num_params_cached(
    lookup_num_params,
    is_moe=is_moe,
    use_kernel=use_kernel,
    init_lr=init_lr,
    max_lr=max_lr,
    decay_alpha=decay_alpha,
    warmup_fraction=warmup_fraction,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    adam_eps=adam_eps,
    grad_clip_norm=grad_clip_norm,
    load_balancing_alpha=load_balancing_alpha,
    num_experts=num_experts,
    top_k=top_k,
    batch_size=batch_size,
    val_num_tokens=val_num_tokens,
    val_seed=val_seed,
    weight_dtype=weight_dtype,
    compute_dtype=compute_dtype,
    BLOCK_B=BLOCK_B,
    BLOCK_D=BLOCK_D,
    BLOCK_F=BLOCK_F,
  ))
  config['requested_num_params'] = int(requested_num_params)
  config['lookup_num_params'] = int(lookup_num_params)
  config['parameter_request_clamped'] = requested_num_params != lookup_num_params
  config['param_target_kind'] = 'num_params'
  config['config_family'] = 'legacy_target_matching' if legacy_configs else 'target_matching'
  config['legacy_configs'] = bool(legacy_configs)
  num_params, activated_params, _, _ = estimate_params_and_flops(config)
  config['num_params'] = int(num_params)
  config['activated_params'] = int(activated_params)
  config['param_error'] = int(num_params - requested_num_params)
  config['param_relative_error'] = float(config['param_error'] / requested_num_params)
  config['mlp_ratio'] = float(config['mlp_dim'] / config['hidden_dim'])
  return config


def transformer_experiment(is_moe):
  ratios = list(range(1, 15))
  token_budgets = [1.0e4, 5.0e4, 1.0e5, 2.5e5, 5.0e5, 7.5e5, 1e6, 1.5e6, 2.0e6]
  configs = [get_config_for_model_size(ratio, is_moe) for ratio in ratios]

  for config in configs:
    for num_tokens in token_budgets:
      config['num_tokens'] = num_tokens
      num_params, activated_params, flops, _ = estimate_params_and_flops(config)
      print(f'Experiment: {num_params:.2e} params, {activated_params:.2e} activated params, {num_tokens:.2e} tokens, {flops:.2e} flops')

      print('Training...')
      # params, train_loss = run_experiment(config, jit=True)

      # we can compute the actual test loss since our test set is fixed and finite
      # but this takes forever
      path = 'dense_results.csv' if not is_moe else 'moe_results.csv'
      # print('Computing test loss...')
      # val_loss = eval_model(params, test_set, 10000, config)
      with open(path, 'a') as f:
        f.write(f'{num_params},{activated_params},{num_tokens},{flops},{-1}\n')

# with open('moe_results.csv', 'a') as f:
#   f.write(f'num_params,activated_params,num_tokens,flops,val_loss\n')
# transformer_experiment(is_moe=True)
