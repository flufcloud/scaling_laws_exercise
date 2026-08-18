from functools import lru_cache

import numpy as np
from jax import numpy as jnp


def create_training_dataset(num_samples, seed=12, rng=None):
  if rng is None:
    rng = np.random.default_rng(seed)

  x = rng.integers(0, 1000, size=int(num_samples), dtype=np.int32)
  y = rng.integers(0, 1000, size=int(num_samples), dtype=np.int32)
  z = x + y
  return jnp.stack((x, y, z), axis=1)

def create_test_dataset():
  x = jnp.arange(1000).repeat(1000, axis=0)
  y = jnp.tile(jnp.arange(1000), 1000)
  z = x + y
  return jnp.stack((x, y, z), axis=1)


@lru_cache(maxsize=1)
def create_all_addition_token_dataset():
  x = np.repeat(np.arange(1000, dtype=np.int32), 1000)
  y = np.tile(np.arange(1000, dtype=np.int32), 1000)
  table = np.stack((x, y, x + y), axis=1)
  return jnp.asarray(tokenize_addition_table(table), dtype=jnp.int32)

def datapoint_to_string(example):
  x = str(example[0])
  y = str(example[1])
  z = str(example[2])

  seq = x + ' + ' + y + ' = ' + z
  return seq.ljust(16)

def tokenizer(seq):
  vocab = {
      '0': 0,
      '1': 1,
      '2': 2,
      '3': 3,
      '4': 4,
      '5': 5,
      '6': 6,
      '7': 7,
      '8': 8,
      '9': 9,
      '+': 10,
      '=': 11,
      ' ': 12,
  }

  return [vocab[c] for c in seq]


def digit_count(values):
  values = np.asarray(values)
  return 1 + (values >= 10).astype(np.int32) + (values >= 100).astype(np.int32) + (values >= 1000).astype(np.int32)


def write_number_tokens(seq, offset, values, max_digits):
  values = np.asarray(values, dtype=np.int32)
  n = values.shape[0]
  counts = digit_count(values)
  divs = (10 ** np.arange(max_digits - 1, -1, -1, dtype=np.int32)).reshape(1, max_digits)
  digits = ((values.reshape(n, 1) // divs) % 10).astype(np.int32)

  cols = np.arange(max_digits, dtype=np.int32).reshape(1, max_digits)
  starts = (max_digits - counts).reshape(n, 1)
  valid = cols >= starts
  dest = offset.reshape(n, 1) + cols - starts

  rows = np.broadcast_to(np.arange(n).reshape(n, 1), (n, max_digits))[valid]
  seq[rows, dest[valid]] = digits[valid]
  return offset + counts


def write_literal_tokens(seq, offset, token_ids):
  rows = np.arange(seq.shape[0])
  for i, token_id in enumerate(token_ids):
    seq[rows, offset + i] = token_id
  return offset + len(token_ids)


def tokenize_addition_table(table):
  table = np.asarray(table, dtype=np.int32)
  x = table[:, 0]
  y = table[:, 1]
  z = table[:, 2]
  seq = np.full((table.shape[0], 16), 12, dtype=np.int32)
  offset = np.zeros(table.shape[0], dtype=np.int32)

  offset = write_number_tokens(seq, offset, x, 3)
  offset = write_literal_tokens(seq, offset, [12, 10, 12])
  offset = write_number_tokens(seq, offset, y, 3)
  offset = write_literal_tokens(seq, offset, [12, 11, 12])
  write_number_tokens(seq, offset, z, 4)
  return seq


def create_training_token_dataset(num_samples, seed=12, rng=None):
  dataset = create_training_dataset(num_samples, seed=seed, rng=rng).tolist()
  dataset = [datapoint_to_string(example) for example in dataset]
  dataset = [tokenizer(example) for example in dataset]
  return jnp.array(np.array(dataset))
