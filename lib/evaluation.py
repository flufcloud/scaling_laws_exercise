from functools import partial
from itertools import combinations_with_replacement

import jax
import numpy as np
from jax import numpy as jnp

from .training import cross_entropy_loss, get_batch
from .transformer import apply_model


def build_digit_error_rows(group_counts):
  rows = []
  for operand_digits_1, operand_digits_2 in combinations_with_replacement(range(1, 4), 2):
    counts = group_counts.get((operand_digits_1, operand_digits_2), {"total": 0, "correct": 0})
    total = counts["total"]
    correct = counts["correct"]
    errors = total - correct
    accuracy_percent = (100 * correct / total) if total else None
    error_percent = (100 * errors / total) if total else None
    rows.append({
      "digit_combination": f"{operand_digits_1}+{operand_digits_2}",
      "operand_digits_1": operand_digits_1,
      "operand_digits_2": operand_digits_2,
      "num_examples": total,
      "num_correct": correct,
      "num_errors": errors,
      "accuracy_percent": accuracy_percent,
      "error_percent": error_percent,
    })
  return rows


def eval_model(params, test_set, print_every, config, return_digit_errors=False):
  def tokens_to_str(tokens):
    vocab = {
      0: '0',
      1: '1',
      2: '2',
      3: '3',
      4: '4',
      5: '5',
      6: '6',
      7: '7',
      8: '8',
      9: '9',
      10: '+',
      11: '=',
      12: ' ',
      100: '',
      }
    return ''.join([vocab[x] for x in tokens])

  def eval_step(batch, params, config,):
    logits, _ = apply_model(params, batch, config)
    aligned_logits = logits[:,:-1]
    aligned_labels = batch[:, 1:]

    return cross_entropy_loss(aligned_logits, aligned_labels, 0, config)

  def eval_step_with_digit_errors(batch, params, config):
    logits, _ = apply_model(params, batch, config)
    aligned_logits = logits[:, :-1]
    aligned_labels = batch[:, 1:]
    predictions = aligned_logits.argmax(axis=-1)

    plus_positions = jnp.argmax(batch == 10, axis=1)
    equals_positions = jnp.argmax(batch == 11, axis=1)
    first_operand_digits = plus_positions - 1
    second_operand_digits = equals_positions - plus_positions - 3

    original_positions = jnp.arange(1, batch.shape[1])
    answer_digit_mask = (
      (original_positions[None, :] > (equals_positions[:, None] + 1))
      & (aligned_labels >= 0)
      & (aligned_labels <= 9)
    )
    correct_answers = jnp.all(
      (predictions == aligned_labels) | ~answer_digit_mask,
      axis=1,
    )

    loss = cross_entropy_loss(aligned_logits, aligned_labels, 0, config)
    return loss, first_operand_digits, second_operand_digits, correct_answers

  def eval_step_with_sample_string(batch, params, config,):
    logits, _ = apply_model(params, batch, config)
    aligned_logits = logits[:,:-1]
    aligned_labels = batch[:, 1:]
    preds = aligned_logits.argmax(axis=-1)

    # mask the preds to after the equal sign so it prints the models predicted answer to the addition
    indices_space_after_eq_pos = jnp.argmax(aligned_labels == 11, axis=1) + 1
    pred_mask = jnp.tile(jnp.arange(preds.shape[1]), (preds.shape[0], 1))
    masked_preds = jnp.where(pred_mask > indices_space_after_eq_pos[:,None], preds, 100) # model prediction for after the equal sign
    preds_strs = [tokens_to_str(X) for X in masked_preds.tolist()]
    targets_strs = [tokens_to_str(X) for X in batch.tolist()]
    eval_string = list(zip(targets_strs, preds_strs))

    # pass in the original because cross-entropy already does masking
    return cross_entropy_loss(aligned_logits, aligned_labels, 0, config), eval_string

  eval_jit = jax.jit(partial(eval_step, config=config))
  eval_with_digit_errors_jit = None
  if return_digit_errors:
    eval_with_digit_errors_jit = jax.jit(partial(eval_step_with_digit_errors, config=config))

  avg_loss = 0
  group_counts = {}
  for i in range(len(test_set) // config['batch_size']):
    batch = get_batch(test_set, i, config['batch_size'])
    if return_digit_errors:
      loss, first_digits, second_digits, correct_answers = eval_with_digit_errors_jit(batch, params)
      for first, second, correct in zip(
        np.asarray(first_digits),
        np.asarray(second_digits),
        np.asarray(correct_answers),
      ):
        digit_pair = tuple(sorted((int(first), int(second))))
        counts = group_counts.setdefault(digit_pair, {"total": 0, "correct": 0})
        counts["total"] += 1
        counts["correct"] += int(correct)
    elif print_every is not None and print_every > 0 and i % print_every == 0:
      loss, _ = eval_step_with_sample_string(batch, params, config)
    else:
      loss = eval_jit(batch, params)

    avg_loss += ((1 / (i + 1))) * (loss - avg_loss)
    if print_every is not None and print_every > 0 and i % print_every == 0:
      print(f"At step {i}/{len(test_set) // config['batch_size']}, average loss was {avg_loss}")

  if return_digit_errors:
    return avg_loss, build_digit_error_rows(group_counts)
  return avg_loss
