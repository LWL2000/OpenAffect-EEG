"""Run frozen five-block coverage diagnostics; writes every result and failure."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.block_validation import block_primary_draws, simulate_blocks
from openaffect_eeg.evaluation_validation import wilson_interval
from openaffect_eeg.interval_diagnostics import intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    design_file = root/cfg['design_source']
    designs = json.loads(design_file.read_text())['datasets']
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.config, design_file, Path(__file__),
               root/'src/openaffect_eeg/block_validation.py',
               root/'src/openaffect_eeg/exposure_statistics.py',
               root/'src/openaffect_eeg/evaluation_validation.py']
    (args.output/'manifest.json').write_text(json.dumps({
        'config': cfg, 'sources': {p.name: sha256_file(p) for p in sources},
        'boundary': 'Test topology and fixed-fit estimator matched; not EEG distribution or refits.'},
        indent=2)+'\n')
    rows, failures = [], []
    started = time.monotonic()
    for di, design in enumerate(designs):
        for scenario in cfg['scenarios']:
            for rep in range(cfg['replicates']):
                run_seed = cfg['seed']+di*10000000+rep
                meta = dict(dataset_id=design['dataset_id'], scenario=scenario, replicate=rep)
                try:
                    table, truth = simulate_blocks(supports=design['supports'], seed=run_seed,
                        scenario=scenario, noise_sd=cfg['noise_sd'], doses=cfg['doses'],
                        target_names=design['target_names'])
                    point, draws, grid, info = block_primary_draws(table,
                        iterations=cfg['bootstrap_iterations'], seed=run_seed+700000,
                        target_names=design['target_names'])
                    if len(draws) < .95*cfg['bootstrap_iterations']:
                        raise ValueError('More than 5% of draws invalid')
                    exact = truth.loc[truth.contrast.eq('eeg_added_matched_calibration')].set_index(
                        ['participant_dose', 'stimulus_dose']).loc[grid, 'true_delta'].to_numpy()
                    # The primary is a mean of paired draws, never a mean of CI endpoints.
                    point = np.append(point, point.mean())
                    draws = np.column_stack([draws, draws.mean(1)])
                    exact = np.append(exact, exact.mean())
                    labels = [f'p{p}_s{s}' for p, s in grid]+['uniform_grid_mean']
                    for method, (low, high) in intervals(point, draws, info['cluster_df']).items():
                        for j, label in enumerate(labels):
                            rows.append(dict(**meta, **info, method=method, estimand=label,
                                truth=exact[j], estimate=point[j], ci_low=low[j], ci_high=high[j],
                                covered=bool(low[j]-1e-12 <= exact[j] <= high[j]+1e-12),
                                positive=bool(low[j]>1e-12)))
                except ValueError as exc:
                    failures.append(dict(**meta, error=str(exc)))
            pd.DataFrame(rows).to_csv(args.output/'replicates.csv', index=False)
            (args.output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
            print(design['dataset_id'], scenario, 'completed', flush=True)
    data = pd.DataFrame(rows)
    summary = []
    for key, group in data.groupby(['dataset_id', 'scenario', 'method', 'estimand']):
        lo, hi = wilson_interval(int(group.covered.sum()), len(group))
        summary.append(dict(zip(['dataset_id', 'scenario', 'method', 'estimand'], key),
            replicates=len(group), coverage=group.covered.mean(), coverage_mc_low=lo,
            coverage_mc_high=hi, bias=(group.estimate-group.truth).mean(),
            positive_rate=group.positive.mean(), mean_width=(group.ci_high-group.ci_low).mean()))
    pd.DataFrame(summary).to_csv(args.output/'summary.csv', index=False)
    (args.output/'completion.json').write_text(json.dumps({
        'status': 'executed_requires_interpretation', 'failures': len(failures),
        'elapsed_seconds': time.monotonic()-started,
        'replicates_sha256': sha256_file(args.output/'replicates.csv')}, indent=2)+'\n')


if __name__ == '__main__':
    main()
