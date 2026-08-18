import json
from functools import lru_cache, partial

import jax
import numpy as np
from jax import numpy as jnp

from .transformer import apply_model, init_params


def init_adam_state(param):
  return {'mu': jnp.zeros_like(param), 'nu': jnp.zeros_like(param), 'count': jnp.array(0)}

def adam_opt_update(adam_beta1, adam_beta2, grad, opt):
  new_opt = {}
  new_opt['mu'] = ((adam_beta1 * opt['mu']) + ((1 - adam_beta1) * grad)).astype(opt['mu'].dtype)
  new_opt['nu'] = ((adam_beta2 * opt['nu']) + ((1 - adam_beta2) * (grad ** 2))).astype(opt['nu'].dtype)
  new_opt['count'] = opt['count'] + 1
  return new_opt

def adam_update(lr, adam_beta1, adam_beta2, adam_eps, param, opt):
  mu_hat = opt['mu'] / (1 - (adam_beta1 ** opt['count']))
  nu_hat = opt['nu'] / (1 - (adam_beta2 ** opt['count']))
  updated = param - ((lr * mu_hat) / (jnp.sqrt(nu_hat) + adam_eps))
  return updated.astype(param.dtype)

def clip_grads_by_global_norm(grads, max_norm):
  if max_norm is None:
    return grads
  squared_norm = sum(jnp.sum(jnp.square(grad.astype(jnp.float32))) for grad in jax.tree.leaves(grads))
  global_norm = jnp.sqrt(squared_norm)
  max_norm = jnp.asarray(max_norm, dtype=jnp.float32)
  scale = jnp.where(max_norm > 0, jnp.minimum(1.0, max_norm / (global_norm + 1e-6)), 1.0)
  return jax.tree.map(lambda grad: grad * scale, grads)

def get_decayed_lr(step, init_lr, max_lr, total_steps, alpha, warmup_fraction, schedule_type="cosine"):
  last_step = jnp.maximum(total_steps - 1, 1)
  warmup_steps = jnp.minimum(
    jnp.floor(total_steps * warmup_fraction),
    jnp.maximum(total_steps - 2, 0),
  )
  warmup_progress = jnp.clip(step / jnp.maximum(warmup_steps, 1), 0.0, 1.0)
  warmup_lr = init_lr + ((max_lr - init_lr) * warmup_progress)

  decay_steps = jnp.maximum(last_step - warmup_steps, 1)
  decay_progress = jnp.clip((step - warmup_steps) / decay_steps, 0.0, 1.0)
  if schedule_type == "constant":
    scheduled_lr = max_lr
  elif schedule_type == "linear":
    scheduled_lr = max_lr * (((1 - alpha) * (1 - decay_progress)) + alpha)
  elif schedule_type == "cosine":
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
    scheduled_lr = max_lr * (((1 - alpha) * cosine_decay) + alpha)
  else:
    raise ValueError(f"unknown schedule_type: {schedule_type}")
  return jnp.where((warmup_steps > 0) & (step < warmup_steps), warmup_lr, scheduled_lr)

def get_batch(ds: jax.Array, step_in_epoch, batch_size):
  pos = step_in_epoch * batch_size
  return ds[pos:pos + batch_size]


def sample_batch(ds: jax.Array, batch_size, rng):
  indices = rng.integers(0, len(ds), size=int(batch_size), dtype=np.int32)
  return ds[indices]

def prediction_cross_entropy_loss(logits, labels, config):
  # we should also mask out the stuff that comes before '=' because we don't want to model the training distribution of possible 3 digit additions

  indices_space_after_eq_pos = jnp.argmax(labels == 11, axis=1) + 1  # gets array of B, that is the position of the '=' in labels

  mask = jnp.tile(jnp.arange(labels.shape[1]), (labels.shape[0], 1))  # creates B x V array where each b is an integer indexing array

  label_mask = jnp.where(mask > indices_space_after_eq_pos[:,None], 1, 0) # whenever mask (B, V) > indices_space_after_eq_pos (B, 1), set to 100, leaves equal sign intact

  # masked is an array with all the labels of each seq before '=' set to 0. This means model learns to predict the equal sign as well (maybe change later)

  logits = logits.astype(jnp.float32)
  one_hots = jax.nn.one_hot(labels, config['vocab_size'], axis=-1).astype(jnp.float32)
  log_losses = jax.nn.log_softmax(logits, axis=-1)
  losses = jnp.sum(log_losses * one_hots, axis=-1)
  losses = losses * label_mask # this should 0 out all the losses for the padded tokens
  # we need to do this manually bc we have arbitrary number of pads per sequence
  final_token_loss = -losses.sum() / label_mask.sum()
  return final_token_loss

def cross_entropy_loss(logits, labels, load_balancing_loss, config):
  return prediction_cross_entropy_loss(logits, labels, config) + load_balancing_loss

def loss_fn(params, batch, config):
  # batch is shape Seq x Pos
  # logits[i] gives the prediction for what the label[i + 1] token will be, and the last token the model predicts has no label
  logits, load_balancing_loss = apply_model(params, batch, config)
  aligned_logits = logits[:, :-1]
  aligned_labels = batch[:, 1:]
  train_loss = prediction_cross_entropy_loss(aligned_logits, aligned_labels, config)
  return train_loss + load_balancing_loss, (train_loss, load_balancing_loss)

def train_step(step, batch, train_state, init_lr, max_lr, decay_alpha, warmup_fraction, total_steps, adam_beta1, adam_beta2, adam_eps, grad_clip_norm, config):
  (loss, (train_loss, load_balancing_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state['params'], batch, config)
  grads = clip_grads_by_global_norm(grads, grad_clip_norm)

  decayed_lr = get_decayed_lr(
    step, init_lr, max_lr, total_steps, decay_alpha, warmup_fraction,
    config.get("schedule_type", "cosine"),
  )
  new_opt = jax.tree.map(partial(adam_opt_update, adam_beta1, adam_beta2), grads, train_state['opt'])
  new_params = jax.tree.map(partial(adam_update, decayed_lr, adam_beta1, adam_beta2, adam_eps), train_state['params'], new_opt)

  return new_params, new_opt, loss, train_loss, load_balancing_loss

def init_train_state(config):
  train_state = {}
  train_state['params'] = init_params(config)
  train_state['opt'] = jax.tree.map(init_adam_state, train_state['params'])
  train_state['lr'] = config['max_lr']
  return train_state


def format_log_value(value):
  if value is None:
    return "None"
  value = float(value)
  if value == 0:
    return "0"
  return f"{value:g}"


STATIC_CONFIG_KEYS = [
  "attention_dim",
  "batch_size",
  "block_sizes",
  "compute_dtype",
  "hidden_dim",
  "load_balancing_alpha",
  "schedule_type",
  "mlp_dim",
  "norm_eps",
  "num_experts",
  "num_kvheads",
  "num_layers",
  "num_qheads",
  "top_k",
  "use_custom_kernel",
  "vocab_size",
  "weight_dtype",
]


def get_static_train_config(config):
  return {key: config[key] for key in STATIC_CONFIG_KEYS if key in config}


@lru_cache(maxsize=128)
def get_cached_train_step(static_config_json):
  static_config = json.loads(static_config_json)
  return jax.jit(partial(train_step, config=static_config))


def get_train_step_func(config, jit):
  if not jit:
    return partial(train_step, config=get_static_train_config(config))

  static_config_json = json.dumps(get_static_train_config(config), sort_keys=True)
  return get_cached_train_step(static_config_json)

def train(train_state, train_set, config, jit):
  train_step_func = get_train_step_func(config, jit)

  steps_per_epoch = int(config['num_tokens']) // config['batch_size']
  total_steps = steps_per_epoch * config['epochs']
  sampling_rng = np.random.default_rng(int(config.get('batch_sampling_seed', config.get('data_seed', 12))))

  for epoch in range(config['epochs']):
    average_train_loss = 0
    for i in range(steps_per_epoch):
      global_step = (epoch * steps_per_epoch) + i
      batch = sample_batch(train_set, config['batch_size'], sampling_rng)
      train_state['params'], train_state['opt'], loss, train_loss, load_balancing_loss = train_step_func(
        global_step,
        batch,
        train_state,
        config.get('init_lr', config['max_lr']),
        config['max_lr'],
        config['decay_alpha'],
        config.get('warmup_fraction', 0.0),
        total_steps,
        config.get('adam_beta1', 0.9),
        config.get('adam_beta2', 0.999),
        config.get('adam_eps', 1e-8),
        config.get('grad_clip_norm'),
      )
      average_train_loss += (1 / (i + 1)) * (train_loss - average_train_loss)

      if i % 5000 == 0:
        log_message = f"At step {global_step}/{total_steps}, average train loss was {format_log_value(average_train_loss)}"
        if config["num_experts"] > 1:
          log_message += f", load balancing loss was {format_log_value(load_balancing_loss)}"
        print(log_message)
  return train_state['params'], average_train_loss
