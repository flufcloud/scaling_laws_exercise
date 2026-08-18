from functools import partial

import jax
from jax import numpy as jnp
from jax.scipy.optimize import minimize


delta = 1e-2

MIN_EXPONENT = 1e-8
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
BETA_MIN = 0.1
BETA_MAX = 1.0


def logit_bounded(value, lower, upper):
  value = jnp.clip(value, lower + MIN_EXPONENT, upper - MIN_EXPONENT)
  p = (value - lower) / (upper - lower)
  p = jnp.clip(p, MIN_EXPONENT, 1 - MIN_EXPONENT)
  return jnp.log(p) - jnp.log1p(-p)


def sigmoid_bounded(raw, lower, upper):
  return lower + ((upper - lower) * jax.nn.sigmoid(raw))


def raw_to_model_params(raw_params, e_upper):
  A = jnp.exp(raw_params[0])
  B = jnp.exp(raw_params[1])
  E = sigmoid_bounded(raw_params[2], 0.0, e_upper)
  alpha = sigmoid_bounded(raw_params[3], ALPHA_MIN, ALPHA_MAX)
  beta = sigmoid_bounded(raw_params[4], BETA_MIN, BETA_MAX)
  return jnp.stack((A, B, E, alpha, beta))


def model_to_raw_params(model_params, e_upper):
  A = jnp.maximum(model_params[0], MIN_EXPONENT)
  B = jnp.maximum(model_params[1], MIN_EXPONENT)
  E = jnp.clip(model_params[2], 0.0, e_upper)
  alpha = jnp.clip(model_params[3], ALPHA_MIN, ALPHA_MAX)
  beta = jnp.clip(model_params[4], BETA_MIN, BETA_MAX)
  return jnp.stack((
    jnp.log(A),
    jnp.log(B),
    logit_bounded(E, 0.0, e_upper),
    logit_bounded(alpha, ALPHA_MIN, ALPHA_MAX),
    logit_bounded(beta, BETA_MIN, BETA_MAX),
  ))


def predict(params, N, D):
  A = params[0]
  B = params[1]
  E = params[2]
  alpha = params[3]
  beta = params[4]

  pred = E + (A / (N ** alpha)) + (B / (D ** beta))
  return pred


def predict_log_loss(raw_params, N, D, e_upper):
  params = raw_to_model_params(raw_params, e_upper)
  A, B, E, alpha, beta = params
  temp_A = jnp.log(A) - (alpha * jnp.log(N))
  temp_B = jnp.log(B) - (beta * jnp.log(D))
  temp_E = jnp.broadcast_to(jnp.log(jnp.maximum(E, MIN_EXPONENT)), temp_A.shape[0])
  return jax.nn.logsumexp(jnp.stack((temp_E, temp_A, temp_B)), axis=0)


def huber_loss(raw_params, delta, N, D, y, e_upper):
  pred = predict_log_loss(raw_params, N, D, e_upper)
  log_y = jnp.log(y)
  error = log_y - pred
  abs_error = jnp.abs(error)
  squared_loss = 0.5 * (error ** 2)
  abs_loss = delta * (abs_error - (delta / 2))
  loss = jnp.where(abs_error <= delta, squared_loss, abs_loss)
  return loss.mean()


def optimize(a, b, e, alpha, beta, N, D, y, e_upper):
  init_model_params = jnp.stack((jnp.exp(a), jnp.exp(b), e, alpha, beta))
  init_raw_params = model_to_raw_params(init_model_params, e_upper)
  minimize_partial = partial(minimize, fun=huber_loss, args=(delta, N, D, y, e_upper), method='BFGS')
  opt = minimize_partial(x0=init_raw_params)

  opt_params = raw_to_model_params(opt.x, e_upper)
  pred = predict(opt_params, N, D)
  log_pred = jnp.log(pred)
  log_y = jnp.log(y)
  max_abs_log_err = jnp.max(jnp.abs(log_pred - log_y))
  max_abs_loss_err = jnp.max(jnp.abs(pred - y))
  return init_model_params, opt_params, opt.fun, max_abs_log_err, max_abs_loss_err


def compute_scaling_law(params):
  A, B, E, alpha, beta = params
  denom = alpha + beta
  G = (alpha * A / (beta * B)) ** (1 / denom)
  a = beta / denom
  b = alpha / denom
  return G, a, b


def N_opt(C, G, a, b):
  return G * ((C / 6) ** a)


def D_opt(C, G, a, b):
  return (G ** -1) * ((C / 6) ** b)


def compute_optimal_nd(params, flops):
  G, a, b = compute_scaling_law(params)
  return N_opt(flops, G, a, b), D_opt(flops, G, a, b)


def interp_loss_for_model(group, target_flops, loss_col="val_loss"):
  import numpy as np

  group = group.sort_values("flops")
  group = group.groupby("flops", as_index=False).agg(**{loss_col: (loss_col, "mean")})

  x = np.log10(group["flops"].to_numpy(dtype=float))
  y = group[loss_col].to_numpy(dtype=float)

  target_x = np.log10(target_flops)

  if target_x < x.min() or target_x > x.max():
    return np.nan

  return np.interp(target_x, x, y)


def fmt_params(x, _):
  if x >= 1e9:
    return f"{x / 1e9:g}B"
  if x >= 1e6:
    return f"{x / 1e6:g}M"
  if x >= 1e3:
    return f"{x / 1e3:g}K"
  return f"{x:g}"


def fmt_flops(x):
  import numpy as np

  exponent = int(np.floor(np.log10(x)))
  mantissa = x / (10 ** exponent)
  if mantissa >= 9.95:
    mantissa = 1
    exponent += 1
  return f"{mantissa:.2g}e{exponent}"


def prepare_isoflop_data(df, loss_col="val_loss"):
  import numpy as np
  import pandas as pd

  cols = ["num_params", "activated_params", "num_tokens", "flops", loss_col]
  missing = [col for col in cols if col not in df.columns]
  if missing:
    raise ValueError(f"Missing required isoFLOP columns: {missing}")

  clean = df.copy()
  for col in cols:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

  clean = clean.dropna(subset=cols)
  for col in cols:
    clean = clean[np.isfinite(clean[col])]
  clean = clean[(clean["num_params"] > 0) & (clean["activated_params"] > 0) & (clean["num_tokens"] > 0)]
  clean = clean[(clean["flops"] > 0) & (clean[loss_col] > 0)]

  if len(clean) == 0:
    raise ValueError("No valid positive isoFLOP rows after cleaning.")

  group_cols = ["num_params", "activated_params", "num_tokens", "flops"]
  clean = (
    clean.groupby(group_cols, as_index=False)
      .agg(**{loss_col: (loss_col, "mean"), "num_runs": (loss_col, "size")})
      .sort_values(["num_params", "activated_params", "num_tokens"])
      .reset_index(drop=True)
  )
  return clean


def load_isoflop_data(results_csv="dense_results.csv", loss_col="val_loss"):
  import pandas as pd

  return prepare_isoflop_data(pd.read_csv(results_csv), loss_col=loss_col)


def build_isoflop_slices(df, min_points_per_curve=5, n_curves=9, loss_col="val_loss"):
  import numpy as np
  import pandas as pd

  if n_curves < 1:
    raise ValueError("n_curves must be >= 1")
  if min_points_per_curve < 1:
    raise ValueError("min_points_per_curve must be >= 1")

  df = prepare_isoflop_data(df, loss_col=loss_col)
  flop_coeffs = df["flops"] / (df["activated_params"] * df["num_tokens"])
  flop_coeff = flop_coeffs.median()
  coeff_rel_error = np.abs((flop_coeffs - flop_coeff) / flop_coeff)

  min_flops = df.groupby("num_params")["flops"].min()
  max_flops = df.groupby("num_params")["flops"].max()
  candidate_flops = np.geomspace(df["flops"].min(), df["flops"].max(), 200)
  valid_counts = np.array([
    ((min_flops <= C) & (max_flops >= C)).sum()
    for C in candidate_flops
  ])
  usable_flops = candidate_flops[valid_counts >= min_points_per_curve]
  if len(usable_flops) == 0:
    raise ValueError("No FLOP budgets have enough model sizes to interpolate.")

  iso_flops = np.geomspace(usable_flops.min(), usable_flops.max(), n_curves)
  rows = []

  for C in iso_flops:
    for num_params, group in df.groupby("num_params"):
      loss = interp_loss_for_model(group, C, loss_col=loss_col)
      if np.isnan(loss):
        continue

      activated_params = group["activated_params"].iloc[0]
      rows.append({
        "iso_flops": C,
        "num_params": num_params,
        "activated_params": activated_params,
        "num_tokens": C / (flop_coeff * activated_params),
        "loss": loss,
      })

  iso = pd.DataFrame(rows)
  if len(iso) == 0:
    raise ValueError("No isoFLOP slices were produced.")

  points_per_curve = iso.groupby("iso_flops").size().to_dict()
  best_points = (
    iso.sort_values(["iso_flops", "loss"])
      .groupby("iso_flops", as_index=False)
      .first()
      .sort_values("iso_flops")
      .reset_index(drop=True)
  )

  diagnostics = {
    "mode": "log_flop_interpolation",
    "num_input_rows": int(len(df)),
    "num_models": int(df[["num_params", "activated_params"]].drop_duplicates().shape[0]),
    "global_min_flops": float(df["flops"].min()),
    "global_max_flops": float(df["flops"].max()),
    "requested_num_curves": int(n_curves),
    "num_curves": int(len(iso_flops)),
    "points_per_curve": {f"{float(k):.6e}": int(v) for k, v in points_per_curve.items()},
    "max_flop_coeff_relative_error": float(coeff_rel_error.max()),
  }

  return iso, iso_flops, flop_coeff, best_points, diagnostics


def plot_isoflop_curves_from_df(
  df,
  output_path="isoflop_curves.png",
  min_points_per_curve=5,
  n_curves=9,
  show=True,
  title="IsoFLOP Curves",
  loss_col="val_loss",
  loss_label="Validation Loss",
):
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FuncFormatter
  import numpy as np

  iso, iso_flops, flop_coeff, best_points, diagnostics = build_isoflop_slices(
    df,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
    loss_col=loss_col,
  )

  plt.figure(figsize=(7.2, 5.6))

  colors = plt.cm.viridis_r(np.linspace(0.12, 0.82, len(iso_flops)))

  plotted_curves = 0
  for color, C in zip(colors, iso_flops):
    s = iso[iso["iso_flops"] == C].sort_values("num_params")

    if len(s) < min_points_per_curve:
      continue
    plotted_curves += 1

    x = s["num_params"].to_numpy(dtype=float)
    y = s["loss"].to_numpy(dtype=float)

    plt.scatter(
      x,
      y,
      s=58,
      color=color,
      zorder=3,
      label=fmt_flops(C),
    )

    plt.plot(
      x,
      y,
      color=color,
      linewidth=2,
      alpha=0.9,
      zorder=2,
    )

    if len(s) >= 4:
      log_x = np.log10(x)
      coeffs = np.polyfit(log_x, y, deg=2)

      x_smooth = np.geomspace(x.min(), x.max(), 200)
      y_smooth = np.polyval(coeffs, np.log10(x_smooth))

      plt.plot(
        x_smooth,
        y_smooth,
        linestyle="--",
        linewidth=1,
        color=color,
        alpha=0.65,
        zorder=1,
      )

  plt.xscale("log")

  plt.xlabel("Parameters", fontsize=13)
  plt.ylabel(loss_label, fontsize=13)
  plt.title(title, fontsize=14)

  plt.gca().xaxis.set_major_formatter(FuncFormatter(fmt_params))

  plt.grid(True, which="both", alpha=0.35)
  plt.legend(title="FLOPs", fontsize=10, title_fontsize=10)

  plt.tight_layout()
  plt.savefig(output_path, dpi=250)
  if show:
    plt.show()
  else:
    plt.close()

  return {
    "output_path": str(output_path),
    "num_rows": int(len(iso)),
    "num_curves": int(plotted_curves),
    "selected_iso_flops": [float(x) for x in iso_flops],
    "flop_coeff": float(flop_coeff),
    "diagnostics": diagnostics,
    "best_points": best_points,
    "iso_slices": iso,
  }


def plot_isoflop_curves(
  results_csv="dense_results.csv",
  output_path="isoflop_curves.png",
  min_points_per_curve=5,
  n_curves=9,
  show=True,
):
  df = load_isoflop_data(results_csv)
  result = plot_isoflop_curves_from_df(
    df,
    output_path=output_path,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
    show=show,
  )
  result.pop("iso_slices", None)
  result.pop("best_points", None)
  return result




def make_json_safe(value):
  from pathlib import Path

  if isinstance(value, dict):
    return {str(k): make_json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [make_json_safe(v) for v in value]
  if hasattr(value, "item"):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  return value


def fit_chinchilla_parameters(df, loss_col="val_loss", results_kind="sweep"):
  import numpy as np
  import pandas as pd

  parameter_col = "num_params" if str(results_kind).lower() in {"total", "moe_total"} else "activated_params"
  cols = [parameter_col, "num_tokens", loss_col]
  fit_df = df.copy()
  for col in cols:
    fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")
  fit_df = fit_df.dropna(subset=cols)
  fit_df = fit_df[fit_df[loss_col] > 0]
  if len(fit_df) == 0:
    raise ValueError("No positive loss rows available for Chinchilla fitting.")

  N = jnp.array(fit_df[parameter_col].to_numpy(dtype=np.float32))
  D = jnp.array(fit_df["num_tokens"].to_numpy(dtype=np.float32))
  y = jnp.array(fit_df[loss_col].to_numpy(dtype=np.float32))

  min_observed_loss = jnp.min(y)
  e_upper = jnp.maximum(min_observed_loss, MIN_EXPONENT)

  a_grid = jnp.array([0., 5., 10., 15., 20.])
  b_grid = jnp.array([0., 5., 10., 15., 20.])
  e_grid = jnp.array([0.5, 0.8, 0.95]) * e_upper
  alpha_grid = jnp.array([0.3])
  beta_grid = jnp.array([0.3])

  vectorized_optimize = jax.vmap(
    jax.vmap(
      jax.vmap(
        jax.vmap(
          jax.vmap(optimize, in_axes=(None, None, None, None, 0, None, None, None, None)),
          in_axes=(None, None, None, 0, None, None, None, None, None),
        ),
        in_axes=(None, None, 0, None, None, None, None, None, None),
      ),
      in_axes=(None, 0, None, None, None, None, None, None, None),
    ),
    in_axes=(0, None, None, None, None, None, None, None, None),
  )
  inits, best_params, fit_losses, max_abs_log_errs, max_abs_loss_errs = jax.jit(vectorized_optimize)(
    a_grid, b_grid, e_grid, alpha_grid, beta_grid, N, D, y, e_upper
  )

  argmin = jnp.nanargmin(fit_losses)
  index = jnp.unravel_index(argmin, fit_losses.shape)
  init_params = inits[index]
  params = best_params[index]
  fit_loss = fit_losses[index]
  max_abs_log_err = max_abs_log_errs[index]
  max_abs_loss_err = max_abs_loss_errs[index]
  y_hat = predict(params, N, D)
  G, chinchilla_a, chinchilla_b = compute_scaling_law(params)

  param_names = ["A", "B", "E", "alpha", "beta"]
  return {
    "results_kind": results_kind,
    "parameter_col": parameter_col,
    "loss_col": loss_col,
    "num_rows": int(len(fit_df)),
    "bounds": {
      "E": [0.0, float(e_upper)],
      "alpha": [ALPHA_MIN, ALPHA_MAX],
      "beta": [BETA_MIN, BETA_MAX],
    },
    "init_params": {name: float(value) for name, value in zip(param_names, init_params.tolist())},
    "best_params": {name: float(value) for name, value in zip(param_names, params.tolist())},
    "scaling_law": {
      "G": float(G),
      "a": float(chinchilla_a),
      "b": float(chinchilla_b),
      "N_opt_formula": "N_opt(C) = G * (C / 6) ** a",
      "D_opt_formula": "D_opt(C) = (1 / G) * (C / 6) ** b",
    },
    "fit_loss": float(fit_loss),
    "max_abs_log_err": float(max_abs_log_err),
    "max_abs_loss_err": float(max_abs_loss_err),
    "max_err": float(max_abs_loss_err),
    "mean_predicted_loss": float(jnp.mean(y_hat)),
  }


def get_sweep_dir(mode, sweep_name, output_root=None):
  from pathlib import Path

  if mode not in {"dense", "moe"}:
    raise ValueError("mode must be dense or moe")
  root = Path(output_root) if output_root is not None else Path(__file__).resolve().parent.parent / "experiments"
  return root / mode / sweep_name


def gaussian_smooth(values, window_length=10):
  import numpy as np

  values = np.asarray(values, dtype=float)
  smoothed = np.full(values.shape, np.nan, dtype=float)
  valid = np.isfinite(values)
  if len(values) == 0 or not valid.any():
    return smoothed

  window_length = max(1, int(window_length))
  half_window = window_length // 2
  sigma = max(window_length / 6.0, 1e-12)

  for i in range(len(values)):
    start = max(0, i - half_window)
    end = min(len(values), start + window_length)
    start = max(0, end - window_length)

    window = values[start:end]
    window_valid = valid[start:end]
    if not window_valid.any():
      continue

    offsets = np.arange(start, end, dtype=float) - i
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights = weights[window_valid]
    weights = weights / weights.sum()
    smoothed[i] = float(np.sum(window[window_valid] * weights))

  return smoothed


def read_run_loss(run_dir, smoothing_window=10):
  import numpy as np
  import pandas as pd

  for loss_path in (run_dir / "loss_curve_tail.csv", run_dir / "loss_curve.csv"):
    if not loss_path.exists():
      continue
    df = pd.read_csv(loss_path)
    if len(df) == 0:
      continue

    if "train_loss" in df.columns:
      losses = pd.to_numeric(df["train_loss"], errors="coerce").to_numpy(dtype=float)
    elif "average_train_loss" in df.columns:
      losses = pd.to_numeric(df["average_train_loss"], errors="coerce").to_numpy(dtype=float)
    else:
      continue

    smoothed = gaussian_smooth(losses, window_length=smoothing_window)
    finite = smoothed[np.isfinite(smoothed)]
    if len(finite) > 0:
      return float(finite[-1])

  return None


HYPERPARAMETER_COLUMNS = [
  "init_lr",
  "max_lr",
  "warmup_fraction",
  "decay_alpha",
  "batch_size",
  "adam_beta1",
  "adam_beta2",
  "adam_eps",
  "grad_clip_norm",
  "load_balancing_alpha",
  "num_experts",
  "top_k",
]


def has_complete_run_artifacts(run_dir):
  return (
    run_dir.is_dir()
    and run_dir.name.startswith("run")
    and (run_dir / "config.json").exists()
    and (run_dir / "metadata.json").exists()
  )


def is_config_dir(path, mode=None):
  if not path.is_dir():
    return False
  if mode is None:
    return path.name.startswith("dense_") or path.name.startswith("moe_")
  return path.name.startswith(f"{mode}_")


def find_run_dirs_for_config(config_dir):
  run_dirs = [path for path in config_dir.iterdir() if has_complete_run_artifacts(path)]
  for child in sorted(path for path in config_dir.iterdir() if path.is_dir() and not path.name.startswith("run")):
    run_dirs.extend(path for path in child.iterdir() if has_complete_run_artifacts(path))
  return sorted(run_dirs, key=lambda path: (str(path.parent), path.name))


def has_config_run_dirs(config_dir):
  if not config_dir.is_dir():
    return False
  return len(find_run_dirs_for_config(config_dir)) > 0


def discover_nested_coordinates(sweep_dir):
  coordinates = []
  for coordinate_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
    if is_config_dir(coordinate_dir) and has_config_run_dirs(coordinate_dir):
      continue
    if any(
      is_config_dir(path) and has_config_run_dirs(path)
      for path in coordinate_dir.iterdir()
      if path.is_dir()
    ):
      coordinates.append(coordinate_dir)
  return coordinates


def collect_sweep_runs_from_dir(mode, sweep_name, sweep_dir, grid_coordinate=None):
  import json
  import pandas as pd

  rows = []
  for config_dir in sorted(path for path in sweep_dir.iterdir() if is_config_dir(path, mode)):
    for run_dir in find_run_dirs_for_config(config_dir):
      config_path = run_dir / "config.json"
      metadata_path = run_dir / "metadata.json"
      if not config_path.exists() or not metadata_path.exists():
        continue

      config = json.loads(config_path.read_text(encoding="utf-8"))
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      loss = metadata.get("val_loss", config.get("val_loss"))
      if loss is None:
        loss = read_run_loss(run_dir)
      if loss is None:
        continue

      row = {
        "mode": mode,
        "sweep_name": sweep_name,
        "grid_coordinate": grid_coordinate,
        "experiment_name": config_dir.name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "max_lr_dir": run_dir.parent.name if run_dir.parent != config_dir else None,
        "num_params": metadata.get("num_params", config.get("num_params")),
        "activated_params": metadata.get("activated_params", config.get("activated_params")),
        "requested_num_params": metadata.get("requested_num_params", config.get("requested_num_params")),
        "lookup_num_params": metadata.get("lookup_num_params", config.get("lookup_num_params")),
        "param_error": metadata.get("param_error", config.get("param_error")),
        "param_relative_error": metadata.get("param_relative_error", config.get("param_relative_error")),
        "num_tokens": metadata.get("num_tokens", config.get("num_tokens")),
        "flops": metadata.get("flops", config.get("flops")),
        "val_loss": loss,
        "init_lr": config.get("init_lr", 0.0),
        "max_lr": config.get("max_lr"),
        "warmup_fraction": config.get("warmup_fraction", 0.0),
        "decay_alpha": config.get("decay_alpha"),
        "adam_beta1": config.get("adam_beta1"),
        "adam_beta2": config.get("adam_beta2"),
        "adam_eps": config.get("adam_eps"),
        "grad_clip_norm": config.get("grad_clip_norm"),
        "load_balancing_alpha": config.get("load_balancing_alpha"),
        "ratio": config.get("ratio"),
        "num_experts": config.get("num_experts"),
        "top_k": config.get("top_k"),
        "hidden_dim": config.get("hidden_dim"),
        "mlp_dim": config.get("mlp_dim"),
        "num_layers": config.get("num_layers"),
      }
      rows.append(row)

  if len(rows) == 0:
    raise ValueError(f"No complete run artifacts found in {sweep_dir}")

  return pd.DataFrame(rows).sort_values(["num_params", "num_tokens", "run_name"]).reset_index(drop=True)


def collect_sweep_runs(mode, sweep_name, output_root=None):
  sweep_dir = get_sweep_dir(mode, sweep_name, output_root)
  if not sweep_dir.exists():
    raise FileNotFoundError(f"Could not find sweep directory: {sweep_dir}")
  return collect_sweep_runs_from_dir(mode, sweep_name, sweep_dir)


def result_df_from_runs(df):
  result_df = (
    df.groupby(["num_params", "activated_params", "num_tokens", "flops"], as_index=False)
      .agg(val_loss=("val_loss", "mean"), num_runs=("val_loss", "size"))
      .sort_values(["num_params", "num_tokens"])
      .reset_index(drop=True)
  )
  return result_df


def hyperparameters_from_df(df):
  hyperparameters = {}
  for col in HYPERPARAMETER_COLUMNS:
    if col not in df.columns:
      continue
    values = df[col].dropna().unique()
    if len(values) == 1:
      value = values[0]
      hyperparameters[col] = value.item() if hasattr(value, "item") else value
    elif len(values) > 1:
      hyperparameters[col] = [
        value.item() if hasattr(value, "item") else value
        for value in sorted(values)
      ]
  return hyperparameters


def write_sweep_analysis_outputs(
  df,
  mode,
  sweep_name,
  sweep_dir,
  min_points_per_curve=5,
  n_curves=9,
  *,
  title=None,
  prefix="",
):
  import json

  result_df = result_df_from_runs(df)

  file_prefix = f"{prefix}_" if prefix else ""
  all_runs_csv = sweep_dir / f"{file_prefix}sweep_runs.csv"
  results_csv = sweep_dir / f"{file_prefix}{mode}_results.csv"
  df.to_csv(all_runs_csv, index=False)
  result_df[["num_params", "activated_params", "num_tokens", "flops", "val_loss", "num_runs"]].to_csv(results_csv, index=False)

  result = {
    "mode": mode,
    "sweep_name": sweep_name,
    "sweep_dir": str(sweep_dir),
    "num_runs": int(len(df)),
    "num_scaling_points": int(len(result_df)),
    "sweep_runs_csv": str(all_runs_csv),
    "results_csv": str(results_csv),
    "hyperparameters": hyperparameters_from_df(df),
  }

  fit = fit_chinchilla_parameters(result_df, loss_col="val_loss", results_kind=mode)
  fit_path = sweep_dir / f"{file_prefix}chinchilla_fit.json"
  fit_path.write_text(json.dumps(make_json_safe(fit), indent=2, sort_keys=True), encoding="utf-8")
  result["chinchilla_fit_json"] = str(fit_path)
  result["chinchilla_fit"] = fit

  isoflop_path = sweep_dir / f"{file_prefix}isoflop_curves.png"
  try:
    isoflop = plot_isoflop_curves_from_df(
      load_isoflop_data(results_csv),
      output_path=isoflop_path,
      min_points_per_curve=min_points_per_curve,
      n_curves=n_curves,
      show=False,
      title=title or f"IsoFLOP Curves: {mode}/{sweep_name}",
    )
    iso_slices = isoflop.pop("iso_slices")
    best_points = isoflop.pop("best_points")
    iso_slices_path = sweep_dir / f"{file_prefix}isoflop_slices.csv"
    best_points_path = sweep_dir / f"{file_prefix}isoflop_best_points.csv"
    iso_slices.to_csv(iso_slices_path, index=False)
    best_points.to_csv(best_points_path, index=False)
    result["isoflop_plot"] = str(isoflop_path)
    result["isoflop_slices_csv"] = str(iso_slices_path)
    result["isoflop_best_points_csv"] = str(best_points_path)
    result["isoflop"] = isoflop
  except ValueError as exc:
    result["isoflop_error"] = str(exc)

  summary_path = sweep_dir / f"{file_prefix}sweep_analysis.json"
  summary_path.write_text(json.dumps(make_json_safe(result), indent=2, sort_keys=True), encoding="utf-8")
  result["summary_json"] = str(summary_path)
  return result


def choose_best_coordinate(coordinate_results):
  candidates = []
  for coordinate in coordinate_results:
    df = coordinate["runs_df"]
    result_df = result_df_from_runs(df)
    candidates.append({
      "coordinate_name": coordinate["coordinate_name"],
      "mean_val_loss": float(result_df["val_loss"].mean()),
      "min_val_loss": float(result_df["val_loss"].min()),
      "max_val_loss": float(result_df["val_loss"].max()),
      "num_scaling_points": int(len(result_df)),
      "num_runs": int(len(df)),
      "hyperparameters": hyperparameters_from_df(df),
      "runs_df": df,
    })
  return sorted(candidates, key=lambda row: (row["mean_val_loss"], row["max_val_loss"]))[0]


def run_nested_sweep_analysis(
  mode,
  sweep_name,
  output_root=None,
  min_points_per_curve=5,
  n_curves=9,
):
  import json

  sweep_dir = get_sweep_dir(mode, sweep_name, output_root)
  coordinate_dirs = discover_nested_coordinates(sweep_dir)
  if len(coordinate_dirs) == 0:
    raise ValueError(f"No nested coordinate sweeps found in {sweep_dir}")

  coordinate_results = []
  summaries = []
  for coordinate_dir in coordinate_dirs:
    coordinate_name = coordinate_dir.name
    nested_sweep_name = f"{sweep_name}/{coordinate_name}"
    df = collect_sweep_runs_from_dir(
      mode,
      nested_sweep_name,
      coordinate_dir,
      grid_coordinate=coordinate_name,
    )
    analysis = write_sweep_analysis_outputs(
      df,
      mode,
      nested_sweep_name,
      coordinate_dir,
      min_points_per_curve=min_points_per_curve,
      n_curves=n_curves,
      title=f"IsoFLOP Curves: {mode}/{nested_sweep_name}",
    )
    coordinate_results.append({
      "coordinate_name": coordinate_name,
      "runs_df": df,
      "analysis": analysis,
    })
    summaries.append({
      "coordinate_name": coordinate_name,
      "sweep_dir": str(coordinate_dir),
      "num_runs": analysis["num_runs"],
      "num_scaling_points": analysis["num_scaling_points"],
      "mean_val_loss": float(result_df_from_runs(df)["val_loss"].mean()),
      "hyperparameters": analysis["hyperparameters"],
      "isoflop_plot": analysis.get("isoflop_plot"),
      "isoflop_error": analysis.get("isoflop_error"),
      "summary_json": analysis.get("summary_json"),
    })

  best = choose_best_coordinate(coordinate_results)
  best_analysis = write_sweep_analysis_outputs(
    best["runs_df"],
    mode,
    sweep_name,
    sweep_dir,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
    title=f"IsoFLOP Curves: {mode}/{sweep_name} best hyperparameters",
    prefix="best",
  )

  best_hyperparameters = {
    "selection_strategy": "lowest mean validation loss across aggregated scaling points",
    "coordinate_name": best["coordinate_name"],
    "mean_val_loss": best["mean_val_loss"],
    "min_val_loss": best["min_val_loss"],
    "max_val_loss": best["max_val_loss"],
    "num_runs": best["num_runs"],
    "num_scaling_points": best["num_scaling_points"],
    "hyperparameters": best["hyperparameters"],
    "best_results_csv": best_analysis["results_csv"],
    "best_isoflop_plot": best_analysis.get("isoflop_plot"),
    "best_isoflop_error": best_analysis.get("isoflop_error"),
  }
  best_path = sweep_dir / "best_hyperparameters.json"
  best_path.write_text(json.dumps(make_json_safe(best_hyperparameters), indent=2, sort_keys=True), encoding="utf-8")

  summary = {
    "mode": mode,
    "sweep_name": sweep_name,
    "sweep_dir": str(sweep_dir),
    "nested": True,
    "num_coordinates": len(coordinate_results),
    "coordinate_summaries": sorted(summaries, key=lambda row: row["mean_val_loss"]),
    "best_hyperparameters_json": str(best_path),
    "best_hyperparameters": best_hyperparameters,
    "best_analysis": best_analysis,
  }
  summary_path = sweep_dir / "nested_sweep_analysis.json"
  summary_path.write_text(json.dumps(make_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
  summary["summary_json"] = str(summary_path)
  return summary


def run_sweep_analysis(
  mode,
  sweep_name,
  output_root=None,
  min_points_per_curve=5,
  n_curves=9,
):
  sweep_dir = get_sweep_dir(mode, sweep_name, output_root)
  if not sweep_dir.exists():
    raise FileNotFoundError(f"Could not find sweep directory: {sweep_dir}")

  if discover_nested_coordinates(sweep_dir):
    return run_nested_sweep_analysis(
      mode,
      sweep_name,
      output_root=output_root,
      min_points_per_curve=min_points_per_curve,
      n_curves=n_curves,
    )

  df = collect_sweep_runs_from_dir(mode, sweep_name, sweep_dir)
  return write_sweep_analysis_outputs(
    df,
    mode,
    sweep_name,
    sweep_dir,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
  )


def format_param_range_value(value):
  value = int(value)
  for suffix, divisor in (("b", 1_000_000_000), ("m", 1_000_000), ("k", 1_000)):
    if value >= divisor and value % divisor == 0:
      return f"{value // divisor}{suffix}"
  return str(value)


def filter_parameter_range(df, min_num_params, max_num_params):
  min_num_params = int(min_num_params)
  max_num_params = int(max_num_params)
  if min_num_params <= 0 or max_num_params < min_num_params:
    raise ValueError("parameter range must satisfy 0 < min_num_params <= max_num_params")
  filtered = df[
    (df["num_params"] >= min_num_params)
    & (df["num_params"] <= max_num_params)
  ].copy()
  if filtered.empty:
    raise ValueError(
      f"No completed scaling points have total parameters between {min_num_params} and {max_num_params}"
    )
  return filtered.reset_index(drop=True)


def run_sweep_range_profile(
  mode,
  sweep_name,
  min_num_params,
  max_num_params,
  output_root=None,
  min_points_per_curve=5,
  n_curves=9,
):
  import json

  sweep_dir = get_sweep_dir(mode, sweep_name, output_root)
  if not sweep_dir.exists():
    raise FileNotFoundError(f"Could not find sweep directory: {sweep_dir}")

  coordinate_dirs = discover_nested_coordinates(sweep_dir)
  selected_coordinate = None
  if coordinate_dirs:
    candidates = []
    for coordinate_dir in coordinate_dirs:
      coordinate_name = coordinate_dir.name
      nested_name = f"{sweep_name}/{coordinate_name}"
      runs = collect_sweep_runs_from_dir(
        mode,
        nested_name,
        coordinate_dir,
        grid_coordinate=coordinate_name,
      )
      try:
        filtered = filter_parameter_range(runs, min_num_params, max_num_params)
      except ValueError:
        continue
      candidates.append({"coordinate_name": coordinate_name, "runs_df": filtered})
    if not candidates:
      raise ValueError("No nested sweep coordinate contains completed runs in the requested parameter range")
    best = choose_best_coordinate(candidates)
    df = best["runs_df"]
    selected_coordinate = best["coordinate_name"]
  else:
    df = filter_parameter_range(
      collect_sweep_runs_from_dir(mode, sweep_name, sweep_dir),
      min_num_params,
      max_num_params,
    )

  profile_name = (
    f"{format_param_range_value(min_num_params)}_to_"
    f"{format_param_range_value(max_num_params)}"
  )
  profile_dir = sweep_dir / "range_profiles" / profile_name
  profile_dir.mkdir(parents=True, exist_ok=True)
  result = write_sweep_analysis_outputs(
    df,
    mode,
    sweep_name,
    profile_dir,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
    title=(
      f"IsoFLOP Profile: {mode}/{sweep_name} "
      f"[{format_param_range_value(min_num_params)}, {format_param_range_value(max_num_params)}]"
    ),
  )
  result.update({
    "profile_kind": "parameter_range",
    "min_num_params": int(min_num_params),
    "max_num_params": int(max_num_params),
    "selected_coordinate": selected_coordinate,
    "profile_dir": str(profile_dir),
  })
  summary_path = profile_dir / "sweep_analysis.json"
  summary_path.write_text(json.dumps(make_json_safe(result), indent=2, sort_keys=True), encoding="utf-8")
  result["summary_json"] = str(summary_path)
  return result
