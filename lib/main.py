import argparse
import json
import sys
from pathlib import Path


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
  sys.path.insert(0, str(PACKAGE_PARENT))

try:
  from scaling_transformer.experiments import (
    estimate_params_and_flops,
    get_config_for_num_params,
    parse_num_params_value,
    run_experiment,
  )
except ModuleNotFoundError:
  from lib.experiments import (
    estimate_params_and_flops,
    get_config_for_num_params,
    parse_num_params_value,
    run_experiment,
  )


def annotate_config_metrics(config):
  num_params, activated_params, flops, _ = estimate_params_and_flops(config)
  config['num_params'] = num_params
  config['activated_params'] = activated_params
  config['flops'] = flops
  return config


def build_config(
  num_params,
  num_tokens,
  is_moe,
  use_kernel,
  batch_size=32,
  val_num_tokens=50000,
  val_seed=13,
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
  sweep_name=None,
  weight_dtype='bf16',
  compute_dtype='bf16',
  legacy_configs=False,
  clamp_config=False,
  schedule_type='cosine',
  replicate_id=0,
):
  config = get_config_for_num_params(
    num_params,
    is_moe=is_moe,
    use_kernel=use_kernel,
    legacy_configs=legacy_configs,
    clamp=clamp_config,
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
  )
  config.pop('param_seed', None)
  config.pop('data_seed', None)
  config['num_tokens'] = num_tokens
  config['schedule_type'] = schedule_type
  config['replicate_id'] = int(replicate_id)
  if sweep_name:
    config['sweep_name'] = sweep_name
  return annotate_config_metrics(config)


def summarize_config(config):
  summary = {
    'num_params': float(config['num_params']),
    'activated_params': float(config['activated_params']),
    'requested_num_params': float(config.get('requested_num_params', config['num_params'])),
    'param_error': float(config.get('param_error', 0)),
    'param_relative_error': float(config.get('param_relative_error', 0)),
    'num_tokens': float(config['num_tokens']),
    'flops': float(config['flops']),
    'weight_dtype': config.get('weight_dtype'),
    'compute_dtype': config.get('compute_dtype'),
    'batch_size': int(config['batch_size']),
    'init_lr': float(config.get('init_lr', config['max_lr'])),
    'max_lr': float(config['max_lr']),
    'decay_alpha': float(config['decay_alpha']),
    'warmup_fraction': float(config.get('warmup_fraction', 0.0)),
    'schedule_type': config.get('schedule_type', 'cosine'),
    'replicate_id': int(config.get('replicate_id', 0)),
  }
  if config.get('val_loss') is not None:
    summary['val_loss'] = float(config['val_loss'])
  if config.get('val_exact_answer_accuracy_percent') is not None:
    summary['val_exact_answer_accuracy_percent'] = float(config['val_exact_answer_accuracy_percent'])
    summary['val_exact_answer_error_percent'] = float(config['val_exact_answer_error_percent'])
  if config.get('sweep_name'):
    summary['sweep_name'] = config['sweep_name']
  return summary


def run_isoflops(
  results_csv='dense_results.csv',
  output_path='isoflop_curves.png',
  min_points_per_curve=5,
  n_curves=9,
  show=True,
):
  try:
    from scaling_transformer.analysis import plot_isoflop_curves
  except ModuleNotFoundError:
    from lib.analysis import plot_isoflop_curves

  return plot_isoflop_curves(
    results_csv,
    output_path=output_path,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
    show=show,
  )


def run_sweep_analysis_command(mode, sweep_name, min_points_per_curve=5, n_curves=9):
  try:
    from scaling_transformer.analysis import run_sweep_analysis
  except ModuleNotFoundError:
    from lib.analysis import run_sweep_analysis

  return run_sweep_analysis(
    mode,
    sweep_name,
    min_points_per_curve=min_points_per_curve,
    n_curves=n_curves,
  )


def run_analysis(
  results_csv,
  metadata_csv,
  results_kind='moe',
  min_points_per_curve=5,
  n_curves=9,
):
  import pandas as pd

  try:
    from scaling_transformer.analysis import (
      fit_chinchilla_parameters,
      load_isoflop_data,
      plot_isoflop_curves_from_df,
    )
  except ModuleNotFoundError:
    from lib.analysis import (
      fit_chinchilla_parameters,
      load_isoflop_data,
      plot_isoflop_curves_from_df,
    )

  dff = pd.read_csv(results_csv)
  df = pd.read_csv(metadata_csv)
  df['val_loss'] = dff['val_loss']
  result = fit_chinchilla_parameters(df, loss_col='val_loss', results_kind=results_kind)

  output_dir = Path(results_csv).resolve().parent
  isoflop_path = output_dir / 'isoflop_curves.png'
  try:
    isoflop = plot_isoflop_curves_from_df(
      load_isoflop_data(results_csv),
      output_path=isoflop_path,
      min_points_per_curve=min_points_per_curve,
      n_curves=n_curves,
      show=False,
      title=f'IsoFLOP Curves: {results_kind}',
    )
    iso_slices = isoflop.pop('iso_slices')
    best_points = isoflop.pop('best_points')
    iso_slices_path = output_dir / 'isoflop_slices.csv'
    best_points_path = output_dir / 'isoflop_best_points.csv'
    iso_slices.to_csv(iso_slices_path, index=False)
    best_points.to_csv(best_points_path, index=False)
    result['isoflop_plot'] = str(isoflop_path)
    result['isoflop_slices_csv'] = str(iso_slices_path)
    result['isoflop_best_points_csv'] = str(best_points_path)
    result['isoflop'] = isoflop
  except ValueError as exc:
    result['isoflop_error'] = str(exc)

  return result


def run_local_experiment(args):
  config = build_config(
    args.num_params,
    args.num_tokens,
    args.model_type == 'moe',
    args.kernel,
    batch_size=args.batch_size,
    val_num_tokens=args.val_num_tokens,
    val_seed=args.val_seed,
    init_lr=args.init_lr,
    max_lr=args.max_lr,
    decay_alpha=args.decay_alpha,
    warmup_fraction=args.warmup_fraction,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_eps=args.adam_eps,
    grad_clip_norm=args.grad_clip_norm,
    load_balancing_alpha=args.load_balancing_alpha,
    num_experts=args.num_experts,
    top_k=args.top_k,
    sweep_name=args.sweep_name,
    weight_dtype=args.weight_dtype,
    compute_dtype=args.compute_dtype,
    legacy_configs=args.legacy_configs,
    clamp_config=args.clamp_config,
    schedule_type=args.schedule_type,
    replicate_id=args.replicate_id,
  )
  print(json.dumps(summarize_config(config), indent=2))
  _, train_loss = run_experiment(config, jit=args.jit)
  result = summarize_config(config)
  result['train_loss'] = float(train_loss)
  print(json.dumps(result, indent=2))


def parse_args():
  parser = argparse.ArgumentParser(description='Run transformer scaling experiments and analysis.')
  subparsers = parser.add_subparsers(dest='command', required=True)

  experiment = subparsers.add_parser('experiment')
  experiment.add_argument('--num-params', type=parse_num_params_value, required=True)
  experiment.add_argument('--num-tokens', type=float, required=True)
  model_type = experiment.add_mutually_exclusive_group(required=True)
  model_type.add_argument('--dense', dest='model_type', action='store_const', const='dense')
  model_type.add_argument('--moe', dest='model_type', action='store_const', const='moe')
  experiment.add_argument('--kernel', action='store_true')
  experiment.add_argument('--legacy-configs', action='store_true')
  experiment.add_argument('--clamp-config', action='store_true')
  experiment.add_argument('--batch-size', type=int, default=32)
  experiment.add_argument('--val-num-tokens', type=int, default=50000)
  experiment.add_argument('--val-seed', type=int, default=13)
  experiment.add_argument('--init-lr', type=float, default=0.0)
  experiment.add_argument('--max-lr', type=float, default=1.0e-5)
  experiment.add_argument('--decay-alpha', type=float, default=0.1)
  experiment.add_argument('--warmup-fraction', type=float, default=0.05)
  experiment.add_argument('--schedule-type', choices=['constant', 'linear', 'cosine'], default='cosine')
  experiment.add_argument('--replicate-id', type=int, default=0)
  experiment.add_argument('--adam-beta1', type=float, default=0.9)
  experiment.add_argument('--adam-beta2', type=float, default=0.99)
  experiment.add_argument('--adam-eps', type=float, default=1e-8)
  experiment.add_argument('--grad-clip-norm', type=float, default=1.0)
  experiment.add_argument('--load-balancing-alpha', type=float, default=0.01)
  experiment.add_argument('--num-experts', type=int, default=4)
  experiment.add_argument('--top-k', type=int, default=2)
  experiment.add_argument('--sweep-name', default=None)
  experiment.add_argument('--weight-dtype', default='bf16')
  experiment.add_argument('--compute-dtype', default='bf16')
  experiment.add_argument('--no-jit', dest='jit', action='store_false')
  experiment.set_defaults(jit=True)

  analysis = subparsers.add_parser('analysis')
  analysis.add_argument('--results-csv', default='moe_results.csv')
  analysis.add_argument('--metadata-csv', default='moe_metadata.csv')
  analysis.add_argument('--results-kind', default='moe')
  analysis.add_argument('--min-points-per-curve', type=int, default=5)
  analysis.add_argument('--n-curves', type=int, default=9)

  isoflops = subparsers.add_parser('isoflops')
  isoflops.add_argument('--results-csv', default='dense_results.csv')
  isoflops.add_argument('--output', default='isoflop_curves.png')
  isoflops.add_argument('--min-points-per-curve', type=int, default=5)
  isoflops.add_argument('--n-curves', type=int, default=9)
  isoflops.add_argument('--no-show', dest='show', action='store_false')
  isoflops.set_defaults(show=True)

  sweep_analysis = subparsers.add_parser('sweep-analysis')
  sweep_analysis.add_argument('--mode', choices=['dense', 'moe'], required=True)
  sweep_analysis.add_argument('--sweep-name', required=True)
  sweep_analysis.add_argument('--min-points-per-curve', type=int, default=5)
  sweep_analysis.add_argument('--n-curves', type=int, default=9)

  return parser.parse_args()


def cli_main():
  args = parse_args()
  if args.command == 'experiment':
    run_local_experiment(args)
  elif args.command == 'analysis':
    result = run_analysis(
      args.results_csv,
      args.metadata_csv,
      args.results_kind,
      args.min_points_per_curve,
      args.n_curves,
    )
    print(json.dumps(result, indent=2))
  elif args.command == 'isoflops':
    result = run_isoflops(
      args.results_csv,
      args.output,
      args.min_points_per_curve,
      args.n_curves,
      args.show,
    )
    print(json.dumps(result, indent=2))
  else:
    result = run_sweep_analysis_command(
      args.mode,
      args.sweep_name,
      args.min_points_per_curve,
      args.n_curves,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
  cli_main()
