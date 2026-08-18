from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
  if hasattr(stream, "reconfigure"):
    stream.reconfigure(encoding="utf-8")

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
  sys.path.insert(0, str(PACKAGE_PARENT))

VALID_PLOT_TYPES = {"loss", "load_balancing"}
VALID_TRACKS = {
  "lr",
  "adam",
  "adam_beta1",
  "adam_beta2",
  "adam_eps",
  "grad_clip_norm",
  "load_balancing_alpha",
  "batch_size",
}
VALID_AGGREGATE_STRATEGIES = {"avg", "best"}


def get_output_root():
  return PACKAGE_PARENT / "experiments"


def format_value(value: Any) -> str:
  if value is None:
    return "None"
  if isinstance(value, float):
    return f"{value:g}"
  return str(value)


def run_sort_key(path: Path) -> tuple[int, str]:
  suffix = path.name[3:] if path.name.startswith("run") else ""
  if suffix.isdigit():
    return int(suffix), path.name
  return 10**12, path.name


def resolve_experiment_dir(config_name: str, sweep_name: str | None = None) -> tuple[str, Path]:
  root = get_output_root()
  candidates: list[tuple[str, Path]] = []

  if config_name.startswith("dense_"):
    modes = ["dense"]
  elif config_name.startswith("moe_"):
    modes = ["moe"]
  else:
    modes = ["dense", "moe"]

  for mode in modes:
    if sweep_name:
      candidates.append((mode, root / mode / sweep_name / config_name))
    candidates.append((mode, root / mode / config_name))

  existing = [(mode, path) for mode, path in candidates if path.exists()]
  if len(existing) == 1:
    return existing[0]
  if len(existing) > 1:
    matches = ", ".join(str(path) for _, path in existing)
    raise ValueError(f"Config name is ambiguous; found matches: {matches}")

  checked = ", ".join(str(path) for _, path in candidates)
  raise FileNotFoundError(f"Could not find config {config_name}. Checked: {checked}")


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def get_run_root(mode: str, experiment_dir: Path, config_name: str, sweep_name: str | None = None) -> Path:
  if not sweep_name:
    return experiment_dir

  root = get_output_root()
  candidates = [
    root / mode / sweep_name / config_name,
    experiment_dir / sweep_name,
  ]
  for run_root in candidates:
    if run_root.exists():
      if not run_root.is_dir():
        raise NotADirectoryError(f"Sweep path is not a directory: {run_root}")
      return run_root

  checked = ", ".join(str(path) for path in candidates)
  raise FileNotFoundError(f"Could not find sweep run directory. Checked: {checked}")


def get_run_dirs(run_root: Path) -> list[Path]:
  run_dirs = [path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("run")]
  for child in sorted(path for path in run_root.iterdir() if path.is_dir() and not path.name.startswith("run")):
    run_dirs.extend(path for path in child.iterdir() if path.is_dir() and path.name.startswith("run"))
  run_dirs = sorted(run_dirs, key=lambda path: (path.parent.name, *run_sort_key(path)))
  if len(run_dirs) == 0:
    raise FileNotFoundError(f"No run directories found in {run_root}")
  return run_dirs


def track_label(config: dict[str, Any], track: str) -> str:
  if track == "lr":
    return (
      f"init_lr={format_value(config.get('init_lr', 0.0))}, "
      f"max_lr={format_value(config.get('max_lr'))}, "
      f"warmup_fraction={format_value(config.get('warmup_fraction', 0.0))}, "
      f"decay_alpha={format_value(config.get('decay_alpha'))}"
    )
  if track == "adam":
    return (
      f"adam_beta1={format_value(config.get('adam_beta1'))}, "
      f"adam_beta2={format_value(config.get('adam_beta2'))}, "
      f"adam_eps={format_value(config.get('adam_eps'))}"
    )
  if track == "adam_beta1":
    return f"adam_beta1={format_value(config.get('adam_beta1'))}"
  if track == "adam_beta2":
    return f"adam_beta2={format_value(config.get('adam_beta2'))}"
  if track == "adam_eps":
    return f"adam_eps={format_value(config.get('adam_eps'))}"
  if track == "grad_clip_norm":
    return f"grad_clip_norm={format_value(config.get('grad_clip_norm'))}"
  if track == "load_balancing_alpha":
    return f"load_balancing_alpha={format_value(config.get('load_balancing_alpha'))}"
  if track == "batch_size":
    return f"batch_size={format_value(config.get('batch_size'))}"
  raise ValueError(f"Unsupported track: {track}")


def important_hparams_label(config: dict[str, Any], include_load_balancing: bool) -> str:
  parts = [
    f"batch_size={format_value(config.get('batch_size'))}",
    f"init_lr={format_value(config.get('init_lr', 0.0))}",
    f"max_lr={format_value(config.get('max_lr'))}",
    f"warmup_fraction={format_value(config.get('warmup_fraction', 0.0))}",
    f"decay_alpha={format_value(config.get('decay_alpha'))}",
    f"adam_beta1={format_value(config.get('adam_beta1'))}",
    f"adam_beta2={format_value(config.get('adam_beta2'))}",
    f"adam_eps={format_value(config.get('adam_eps'))}",
    f"grad_clip_norm={format_value(config.get('grad_clip_norm'))}",
  ]
  if include_load_balancing:
    parts.append(f"load_balancing_alpha={format_value(config.get('load_balancing_alpha'))}")
  return ", ".join(parts)


def aggregate_curves(curves, value_col: str, aggregate_strategy: str, include_load_balancing: bool):
  import pandas as pd

  if aggregate_strategy not in VALID_AGGREGATE_STRATEGIES:
    raise ValueError(f"--aggregate-strategy must be one of {sorted(VALID_AGGREGATE_STRATEGIES)}")

  groups = {}
  for item in curves:
    groups.setdefault(item["label"], []).append(item)

  aggregated = []
  for label, items in groups.items():
    if aggregate_strategy == "avg":
      pieces = []
      for item in items:
        run_df = item["df"][["step", value_col]].copy().rename(columns={value_col: item["run_name"]})
        pieces.append(run_df.set_index("step"))
      merged = pd.concat(pieces, axis=1, join="inner").sort_index()
      if len(merged) == 0:
        continue
      mean_values = merged.mean(axis=1)
      legend = f"{label} (n={len(items)})" if len(items) > 1 else label
      aggregated.append({
        "legend": legend,
        "num_runs": len(items),
        "selected_run": None,
        "steps": mean_values.index.to_numpy(),
        "values": mean_values.to_numpy(),
      })
    else:
      best_item = min(items, key=lambda item: float(item["df"][value_col].iloc[-1]))
      df = best_item["df"].sort_values("step")
      legend = f"{best_item['run_name']}: {label}; {important_hparams_label(best_item['config'], include_load_balancing)}"
      if len(items) > 1:
        legend += f" (best of n={len(items)})"
      aggregated.append({
        "legend": legend,
        "num_runs": len(items),
        "selected_run": best_item["run_name"],
        "steps": df["step"].to_numpy(),
        "values": df[value_col].to_numpy(),
      })
  return aggregated


def plot_loss(experiment_name: str, experiment_dir: Path, run_dirs: list[Path], track: str, output_path: Path, aggregate_strategy: str) -> dict[str, Any]:
  import matplotlib.pyplot as plt
  import pandas as pd

  curves = []
  skipped = 0
  plt.figure(figsize=(9, 5.5))

  for run_dir in run_dirs:
    config_path = run_dir / "config.json"
    loss_path = run_dir / "loss_curve.csv"
    if not loss_path.exists():
      print(f"Skipping {run_dir.name}: missing loss_curve.csv")
      skipped += 1
      continue
    if not config_path.exists():
      print(f"Skipping {run_dir.name}: missing config.json")
      skipped += 1
      continue

    config = load_json(config_path)
    df = pd.read_csv(loss_path)
    if len(df) == 0:
      print(f"Skipping {run_dir.name}: empty loss_curve.csv")
      skipped += 1
      continue
    if "step" not in df.columns or "train_loss" not in df.columns:
      raise ValueError(f"{loss_path} must contain step and train_loss columns")

    run_label = f"{run_dir.parent.name}/{run_dir.name}" if run_dir.parent != experiment_dir else run_dir.name
    curves.append({"label": track_label(config, track), "run_name": run_label, "config": config, "df": df})

  aggregated = aggregate_curves(curves, "train_loss", aggregate_strategy, include_load_balancing=False)
  for item in aggregated:
    plt.plot(item["steps"], item["values"], linewidth=1.6, label=item["legend"])

  if len(aggregated) == 0:
    raise ValueError(f"No loss curves could be plotted from {experiment_dir}")

  plt.xlabel("Step")
  plt.ylabel("Train Loss")
  plt.title(f"Train Loss Curves: {experiment_name}")
  plt.grid(True, alpha=0.35)
  plt.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(output_path, dpi=220)
  plt.close()
  return {
    "num_runs": len(curves),
    "num_curves": len(aggregated),
    "num_skipped": skipped,
    "selected_runs": [item["selected_run"] for item in aggregated if item["selected_run"] is not None],
  }


def plot_load_balancing(experiment_name: str, experiment_dir: Path, run_dirs: list[Path], track: str, output_path: Path, aggregate_strategy: str) -> dict[str, Any]:
  import matplotlib.pyplot as plt
  import pandas as pd

  curves = []
  skipped = 0
  title_suffix = ""
  plt.figure(figsize=(9, 5.5))

  for run_dir in run_dirs:
    config_path = run_dir / "config.json"
    load_balancing_path = run_dir / "load_balancing.csv"
    if not load_balancing_path.exists():
      print(f"Skipping {run_dir.name}: missing load_balancing.csv")
      skipped += 1
      continue
    if not config_path.exists():
      print(f"Skipping {run_dir.name}: missing config.json")
      skipped += 1
      continue

    config = load_json(config_path)
    df = pd.read_csv(load_balancing_path)
    if len(df) == 0:
      print(f"Skipping {run_dir.name}: empty load_balancing.csv")
      skipped += 1
      continue
    if "step" not in df.columns or "ratio" not in df.columns:
      raise ValueError(f"{load_balancing_path} must contain step and ratio columns")

    if not title_suffix:
      title_suffix = f" (E={format_value(config.get('num_experts'))}, top_k={format_value(config.get('top_k'))})"

    run_label = f"{run_dir.parent.name}/{run_dir.name}" if run_dir.parent != experiment_dir else run_dir.name
    curves.append({"label": track_label(config, track), "run_name": run_label, "config": config, "df": df})

  aggregated = aggregate_curves(curves, "ratio", aggregate_strategy, include_load_balancing=True)
  for item in aggregated:
    plt.plot(item["steps"], item["values"], linewidth=1.6, label=item["legend"])

  if len(aggregated) == 0:
    raise ValueError(f"No load balancing curves could be plotted from {experiment_dir}")

  plt.xlabel("Step")
  plt.ylabel("Max Expert Assignment Ratio")
  plt.title(f"Load Balancing Curves: {experiment_name}{title_suffix}")
  plt.ylim(bottom=0)
  plt.grid(True, alpha=0.35)
  plt.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(output_path, dpi=220)
  plt.close()
  return {
    "num_runs": len(curves),
    "num_curves": len(aggregated),
    "num_skipped": skipped,
    "selected_runs": [item["selected_run"] for item in aggregated if item["selected_run"] is not None],
  }


def get_sweep_config_dirs(mode: str, sweep_name: str) -> list[Path]:
  _, config_groups = get_sweep_config_groups(mode, sweep_name)
  return [path for paths in config_groups.values() for path in paths]


def get_sweep_config_groups(mode: str, sweep_name: str) -> tuple[Path, dict[str, list[Path]]]:
  if mode not in {"dense", "moe"}:
    raise ValueError("--mode must be dense or moe")

  sweep_dir = get_output_root() / mode / sweep_name
  if not sweep_dir.exists():
    raise FileNotFoundError(f"Could not find sweep directory: {sweep_dir}")
  if not sweep_dir.is_dir():
    raise NotADirectoryError(f"Sweep path is not a directory: {sweep_dir}")

  direct_config_dirs = sorted(
    path for path in sweep_dir.iterdir()
    if path.is_dir() and path.name.startswith(f"{mode}_")
  )
  config_groups = {path.name: [path] for path in direct_config_dirs}

  for coordinate_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
    if coordinate_dir.name.startswith(f"{mode}_"):
      continue
    for config_dir in sorted(
      path for path in coordinate_dir.iterdir()
      if path.is_dir() and path.name.startswith(f"{mode}_")
    ):
      config_groups.setdefault(config_dir.name, []).append(config_dir)

  if len(config_groups) == 0:
    raise FileNotFoundError(f"No config directories found in {sweep_dir}")
  return sweep_dir, dict(sorted(config_groups.items()))


def create_sweep_plots(plot_type: str, mode: str, sweep_name: str, track: str, aggregate_strategy: str = "avg") -> dict[str, Any]:
  if plot_type not in VALID_PLOT_TYPES:
    raise ValueError(f"--type must be one of {sorted(VALID_PLOT_TYPES)}")
  if track not in VALID_TRACKS:
    raise ValueError(f"--track must be one of {sorted(VALID_TRACKS)}")
  if aggregate_strategy not in VALID_AGGREGATE_STRATEGIES:
    raise ValueError(f"--aggregate-strategy must be one of {sorted(VALID_AGGREGATE_STRATEGIES)}")
  if plot_type == "load_balancing" and mode != "moe":
    raise ValueError("--type load_balancing only works for MoE sweeps")

  results = []
  errors = []
  sweep_dir, config_groups = get_sweep_config_groups(mode, sweep_name)
  output_dir = sweep_dir / "plots"
  output_dir.mkdir(parents=True, exist_ok=True)

  for config_name, config_dirs in config_groups.items():
    try:
      if len(config_dirs) == 1 and config_dirs[0].parent == sweep_dir:
        results.append(create_plot(plot_type, config_name, track, sweep_name, aggregate_strategy))
        continue

      run_dirs = []
      for config_dir in config_dirs:
        run_dirs.extend(get_run_dirs(config_dir))

      output_path = output_dir / f"{config_name}_{plot_type}_by_{track}_{aggregate_strategy}.png"
      plot_name = f"{config_name}/{sweep_name}"
      if plot_type == "loss":
        plot_stats = plot_loss(plot_name, sweep_dir, run_dirs, track, output_path, aggregate_strategy)
      else:
        plot_stats = plot_load_balancing(plot_name, sweep_dir, run_dirs, track, output_path, aggregate_strategy)

      results.append({
        "type": plot_type,
        "config": config_name,
        "track": track,
        "aggregate_strategy": aggregate_strategy,
        "sweep_name": sweep_name,
        "mode": mode,
        "num_runs": plot_stats["num_runs"],
        "num_curves": plot_stats["num_curves"],
        "num_skipped": plot_stats["num_skipped"],
        "selected_runs": plot_stats.get("selected_runs", []),
        "run_root": str(sweep_dir),
        "output_path": str(output_path),
      })
    except Exception as exc:
      errors.append({"config": config_name, "error": str(exc)})

  return {
    "type": plot_type,
    "mode": mode,
    "sweep_name": sweep_name,
    "track": track,
    "aggregate_strategy": aggregate_strategy,
    "num_configs": len(results),
    "num_errors": len(errors),
    "output_dir": str(output_dir),
    "results": results,
    "errors": errors,
  }


def create_plot(plot_type: str, config_name: str, track: str, sweep_name: str | None = None, aggregate_strategy: str = "avg") -> dict[str, Any]:
  if plot_type not in VALID_PLOT_TYPES:
    raise ValueError(f"--type must be one of {sorted(VALID_PLOT_TYPES)}")
  if track not in VALID_TRACKS:
    raise ValueError(f"--track must be one of {sorted(VALID_TRACKS)}")
  if aggregate_strategy not in VALID_AGGREGATE_STRATEGIES:
    raise ValueError(f"--aggregate-strategy must be one of {sorted(VALID_AGGREGATE_STRATEGIES)}")

  mode, experiment_dir = resolve_experiment_dir(config_name, sweep_name)
  if plot_type == "load_balancing" and mode != "moe":
    raise ValueError("--type load_balancing only works for MoE experiment configs")

  run_root = get_run_root(mode, experiment_dir, config_name, sweep_name)
  run_dirs = get_run_dirs(run_root)
  output_path = run_root / f"{plot_type}_by_{track}_{aggregate_strategy}.png"
  plot_name = config_name if not sweep_name else f"{config_name}/{sweep_name}"

  if plot_type == "loss":
    plot_stats = plot_loss(plot_name, run_root, run_dirs, track, output_path, aggregate_strategy)
  else:
    plot_stats = plot_load_balancing(plot_name, run_root, run_dirs, track, output_path, aggregate_strategy)

  return {
    "type": plot_type,
    "config": config_name,
    "track": track,
    "aggregate_strategy": aggregate_strategy,
    "sweep_name": sweep_name,
    "mode": mode,
    "num_runs": plot_stats["num_runs"],
    "num_curves": plot_stats["num_curves"],
    "num_skipped": plot_stats["num_skipped"],
    "selected_runs": plot_stats.get("selected_runs", []),
    "run_root": str(run_root),
    "output_path": str(output_path),
  }


def main(
  type: str,
  track: str,
  config: str = "",
  sweep_name: str = "",
  mode: str = "",
  aggregate_strategy: str = "avg",
) -> None:
  if config:
    result = create_plot(type, config, track, sweep_name or None, aggregate_strategy)
  else:
    if not sweep_name or not mode:
      raise ValueError("Either pass --config, or pass both --mode and --sweep-name to plot every config in a sweep.")
    result = create_sweep_plots(type, mode, sweep_name, track, aggregate_strategy)
  print(json.dumps(result, indent=2))

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Create loss or load-balancing plots from local experiment artifacts.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--type", choices=sorted(VALID_PLOT_TYPES), required=True)
  parser.add_argument("--track", choices=sorted(VALID_TRACKS), required=True)
  parser.add_argument("--config", default="", help="Single config folder to plot.")
  parser.add_argument("--sweep-name", default="", help="Sweep folder name. With --mode, plots every config in the sweep.")
  parser.add_argument("--mode", choices=["", "dense", "moe"], default="", help="Experiment mode for plotting every config in a sweep.")
  parser.add_argument("--aggregate-strategy", choices=sorted(VALID_AGGREGATE_STRATEGIES), default="avg")
  return parser.parse_args()


def cli_main() -> None:
  args = parse_args()
  if args.config:
    result = create_plot(args.type, args.config, args.track, args.sweep_name or None, args.aggregate_strategy)
  else:
    if not args.sweep_name or not args.mode:
      raise ValueError("Either pass --config, or pass both --mode and --sweep-name to plot every config in a sweep.")
    result = create_sweep_plots(args.type, args.mode, args.sweep_name, args.track, args.aggregate_strategy)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  cli_main()
