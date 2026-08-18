from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
  sys.path.insert(0, str(PACKAGE_PARENT))

try:
  from scaling_transformer.experiments import estimate_params_and_flops, get_config_for_num_params, parse_num_params_value, supported_num_param_targets
except ModuleNotFoundError:
  from lib.experiments import estimate_params_and_flops, get_config_for_num_params, parse_num_params_value, supported_num_param_targets

FLOP_SUFFIXES = {
  "k": 1e3,
  "m": 1e6,
  "b": 1e9,
  "t": 1e12,
  "p": 1e15,
}


def parse_flops(value: str) -> float:
  text = str(value).strip().lower().replace("_", "")
  if text[-1:] in FLOP_SUFFIXES:
    return float(text[:-1]) * FLOP_SUFFIXES[text[-1]]
  return float(text)


def format_flops(value: float) -> str:
  if value == 0:
    return "0"
  exponent = int(math.floor(math.log10(abs(value))))
  mantissa = value / (10 ** exponent)
  return f"{mantissa:.4g}e{exponent}"


def round_tokens(value: float, multiple: int) -> int:
  if multiple <= 1:
    return max(1, int(round(value)))
  return max(multiple, int(round(value / multiple)) * multiple)


def choose_spread(candidates: list[dict[str, Any]], num_settings: int) -> list[dict[str, Any]]:
  if len(candidates) <= num_settings:
    return candidates
  if num_settings == 1:
    return [candidates[len(candidates) // 2]]

  selected = []
  seen = set()
  last = len(candidates) - 1
  for i in range(num_settings):
    idx = int(round(i * last / (num_settings - 1)))
    while idx in seen and idx + 1 < len(candidates):
      idx += 1
    while idx in seen and idx - 1 >= 0:
      idx -= 1
    seen.add(idx)
    selected.append(candidates[idx])
  return sorted(selected, key=lambda row: row["num_params"])


def choose_log_spread(candidates: list[dict[str, Any]], num_settings: int) -> list[dict[str, Any]]:
  if len(candidates) <= num_settings:
    return candidates
  if num_settings == 1:
    target = math.sqrt(candidates[0]["requested_num_params"] * candidates[-1]["requested_num_params"])
    return [min(candidates, key=lambda row: abs(math.log(row["requested_num_params"] / target)))]

  log_min = math.log(candidates[0]["requested_num_params"])
  log_max = math.log(candidates[-1]["requested_num_params"])
  available = set(range(len(candidates)))
  selected_indexes = []
  for index in range(num_settings):
    target = log_min + (index * (log_max - log_min) / (num_settings - 1))
    selected = min(
      available,
      key=lambda candidate_index: (
        abs(math.log(candidates[candidate_index]["requested_num_params"]) - target),
        candidate_index,
      ),
    )
    available.remove(selected)
    selected_indexes.append(selected)
  return [candidates[index] for index in sorted(selected_indexes)]


def build_candidate(
  requested_num_params: int,
  target_flops: float,
  is_moe: bool,
  batch_size: int,
  num_experts: int,
  top_k: int,
  token_multiple: int,
  legacy_configs: bool = False,
  clamp_configs: bool = False,
) -> dict[str, Any]:
  config = get_config_for_num_params(
    requested_num_params,
    is_moe=is_moe,
    batch_size=batch_size,
    num_experts=num_experts,
    top_k=top_k,
    legacy_configs=legacy_configs,
    clamp=clamp_configs,
  )
  config["num_tokens"] = 1
  num_params, activated_params, _, _ = estimate_params_and_flops(config)

  target_tokens = target_flops / (6 * activated_params)
  num_tokens = round_tokens(target_tokens, token_multiple)

  config["num_tokens"] = num_tokens
  num_params, activated_params, actual_flops, _ = estimate_params_and_flops(config)
  rel_error = (actual_flops - target_flops) / target_flops

  return {
    "requested_num_params": int(requested_num_params),
    "lookup_num_params": int(config["lookup_num_params"]),
    "parameter_request_clamped": bool(config.get("parameter_request_clamped", False)),
    "num_tokens": int(num_tokens),
    "target_tokens": float(target_tokens),
    "num_params": int(num_params),
    "activated_params": int(activated_params),
    "flops": float(actual_flops),
    "flops_pretty": format_flops(float(actual_flops)),
    "relative_error": float(rel_error),
    "relative_error_pct": float(100 * rel_error),
  }


def solve_settings(
  target_flops: float,
  num_settings: int = 10,
  min_num_params: int = 100000,
  max_num_params: int = 10000000,
  param_interval: int = 10000,
  is_moe: bool = False,
  batch_size: int = 32,
  num_experts: int = 4,
  top_k: int = 2,
  token_multiple: int | None = None,
  selection: str = "spread",
  legacy_configs: bool = False,
  clamp_configs: bool = False,
) -> list[dict[str, Any]]:
  if target_flops <= 0:
    raise ValueError("target FLOPs must be positive")
  if min_num_params < 1 or max_num_params < min_num_params:
    raise ValueError("parameter bounds must satisfy 1 <= min_num_params <= max_num_params")
  lower_bound = 10000 if legacy_configs else 100000
  upper_bound = 40000000 if legacy_configs else 500000000
  if not clamp_configs and (min_num_params < lower_bound or max_num_params > upper_bound):
    raise ValueError(
      f"parameter bounds must stay within {lower_bound} to {upper_bound} unless clamp_configs is enabled"
    )
  if param_interval < 1:
    raise ValueError("param_interval must be >= 1")
  if num_settings < 1:
    raise ValueError("num_settings must be >= 1")

  multiple = batch_size if token_multiple is None else token_multiple
  if clamp_configs:
    if num_settings == 1:
      supported = [int(round(math.sqrt(min_num_params * max_num_params)))]
    else:
      log_min, log_max = math.log(min_num_params), math.log(max_num_params)
      supported = [
        int(round(math.exp(log_min + i * (log_max - log_min) / (num_settings - 1))))
        for i in range(num_settings)
      ]
    if len(set(supported)) != len(supported):
      raise ValueError("requested range is too narrow to produce distinct log-spaced parameter targets")
  elif legacy_configs:
    supported = list(range(min_num_params, max_num_params + 1, param_interval))
  else:
    supported = [
      value for value in supported_num_param_targets(is_moe=is_moe)
      if min_num_params <= value <= max_num_params
    ]
  if not supported:
    raise ValueError("No model sizes fall within the requested parameter bounds")

  if selection in {"spread", "log"} and not clamp_configs:
    target_rows = [
      {"requested_num_params": value, "num_params": value}
      for value in supported
    ]
    if selection == "spread":
      target_rows = choose_spread(target_rows, num_settings)
    else:
      target_rows = choose_log_spread(target_rows, num_settings)
    supported = [row["requested_num_params"] for row in target_rows]

  candidates = [
    build_candidate(
      requested_num_params,
      target_flops,
      is_moe,
      batch_size,
      num_experts,
      top_k,
      multiple,
      legacy_configs,
      clamp_configs,
    )
    for requested_num_params in supported
  ]

  if clamp_configs:
    return candidates
  if selection in {"spread", "log"}:
    return candidates
  if selection == "closest":
    selected = sorted(candidates, key=lambda row: abs(row["relative_error"]))[:num_settings]
    return sorted(selected, key=lambda row: row["num_params"])
  raise ValueError("selection must be spread, log, or closest")


def solve_explicit_settings(
  target_flops: float,
  num_param_values: list[int],
  is_moe: bool = False,
  batch_size: int = 32,
  num_experts: int = 4,
  top_k: int = 2,
  token_multiple: int | None = None,
  legacy_configs: bool = False,
  clamp_configs: bool = False,
) -> list[dict[str, Any]]:
  if target_flops <= 0:
    raise ValueError("target FLOPs must be positive")
  if len(num_param_values) == 0:
    raise ValueError("num_param_values must contain at least one model size")

  requested_values = [parse_num_params_value(value) for value in num_param_values]
  if len(set(requested_values)) != len(requested_values):
    raise ValueError("num_param_values must not contain duplicates")
  lower_bound = 10000 if legacy_configs else 100000
  upper_bound = 40000000 if legacy_configs else 500000000
  if not clamp_configs and (min(requested_values) < lower_bound or max(requested_values) > upper_bound):
    raise ValueError(f"num_param_values must be between {lower_bound} and {upper_bound}")

  multiple = batch_size if token_multiple is None else token_multiple
  rows = [
    build_candidate(
      requested_num_params,
      target_flops,
      is_moe,
      batch_size,
      num_experts,
      top_k,
      multiple,
      legacy_configs,
      clamp_configs,
    )
    for requested_num_params in requested_values
  ]
  resolved = [row["num_params"] for row in rows]
  if not clamp_configs and len(set(resolved)) != len(resolved):
    raise ValueError(
      "num_param_values resolve to duplicate architectures for the selected config family; "
      "request distinct lookup targets instead"
    )
  return rows


def command_for_row(row: dict[str, Any], args: argparse.Namespace) -> str:
  mode = "--moe" if args.moe else "--dense"
  parts = [
    "python -m lib.main experiment",
    mode,
    f"--num-params {row['requested_num_params']}",
    f"--num-tokens {row['num_tokens']}",
    f"--batch-size {args.batch_size}",
  ]
  if args.moe:
    parts.extend([f"--num-experts {args.num_experts}", f"--top-k {args.top_k}"])
  if args.init_lr is not None:
    parts.append(f"--init-lr {args.init_lr:g}")
  if args.max_lr is not None:
    parts.append(f"--max-lr {args.max_lr:g}")
  if args.decay_alpha is not None:
    parts.append(f"--decay-alpha {args.decay_alpha:g}")
  if args.warmup_fraction is not None:
    parts.append(f"--warmup-fraction {args.warmup_fraction:g}")
  if args.grad_clip_norm is not None:
    parts.append(f"--grad-clip-norm {args.grad_clip_norm:g}")
  if args.sweep_name:
    parts.append(f"--sweep-name {args.sweep_name}")
  if args.legacy_configs:
    parts.append("--legacy-configs")
  return " ".join(parts)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Solve num_params/num_tokens settings near a target FLOPs budget.")
  parser.add_argument("--flops", required=True, type=parse_flops, help="Target FLOPs, e.g. 1e12, 500b, 2.5t.")
  parser.add_argument("--num-settings", type=int, default=10)
  parser.add_argument("--min-num-params", type=parse_num_params_value, default=100000)
  parser.add_argument("--max-num-params", type=parse_num_params_value, default=40000000)
  parser.add_argument("--param-interval", type=parse_num_params_value, default=10000)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--token-multiple", type=int, default=0, help="Round num_tokens to this multiple. Default: batch size.")
  parser.add_argument("--selection", choices=["spread", "log", "closest"], default="spread")
  parser.add_argument("--legacy-configs", action="store_true", help="Use the former 10K-grid target-matching architecture family.")
  parser.add_argument("--clamp-configs", action="store_true", help="Generate requested targets across the full range and clamp each to the nearest available static architecture.")
  model = parser.add_mutually_exclusive_group()
  model.add_argument("--dense", dest="moe", action="store_false")
  model.add_argument("--moe", dest="moe", action="store_true")
  parser.set_defaults(moe=False)
  parser.add_argument("--num-experts", type=int, default=4)
  parser.add_argument("--top-k", type=int, default=2)
  parser.add_argument("--init-lr", type=float, default=0.0)
  parser.add_argument("--max-lr", type=float, default=None)
  parser.add_argument("--decay-alpha", type=float, default=0.1)
  parser.add_argument("--warmup-fraction", type=float, default=0.05)
  parser.add_argument("--grad-clip-norm", type=float, default=1.0)
  parser.add_argument("--sweep-name", type=str, default="")
  parser.add_argument("--format", choices=["json", "commands", "csv"], default="json")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  rows = solve_settings(
    target_flops=args.flops,
    num_settings=args.num_settings,
    min_num_params=args.min_num_params,
    max_num_params=args.max_num_params,
    param_interval=args.param_interval,
    is_moe=args.moe,
    batch_size=args.batch_size,
    num_experts=args.num_experts,
    top_k=args.top_k,
    token_multiple=args.token_multiple or None,
    selection=args.selection,
    legacy_configs=args.legacy_configs,
    clamp_configs=args.clamp_configs,
  )

  if args.format == "commands":
    for row in rows:
      print(command_for_row(row, args))
    return

  if args.format == "csv":
    print("requested_num_params,num_tokens,num_params,activated_params,flops,relative_error_pct")
    for row in rows:
      print(
        f"{row['requested_num_params']},{row['num_tokens']},{row['num_params']},"
        f"{row['activated_params']},{row['flops']:.0f},{row['relative_error_pct']:.6f}"
      )
    return

  result = {
    "target_flops": float(args.flops),
    "target_flops_pretty": format_flops(float(args.flops)),
    "mode": "moe" if args.moe else "dense",
    "batch_size": args.batch_size,
    "token_multiple": args.token_multiple or args.batch_size,
    "selection": args.selection,
    "config_family": "legacy_target_matching" if args.legacy_configs else "compound",
    "clamp_configs": args.clamp_configs,
    "settings": rows,
    "commands": [command_for_row(row, args) for row in rows],
  }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
