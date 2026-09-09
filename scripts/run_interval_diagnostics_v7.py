"""Run a fresh-seed, all-candidate study without changing production intervals."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.evaluation_validation import simulate_oracle, wilson_interval
from openaffect_eeg.interval_diagnostics import METHODS, intervals, paired_primary_draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if tuple(cfg["methods"]) != METHODS:
        raise ValueError("Candidate set changed")
    root = Path(__file__).resolve().parents[1]
    sources = [args.config, Path(__file__), *[root/"src/openaffect_eeg"/name for name in
               ("interval_diagnostics.py", "evaluation_validation.py", "exposure_statistics.py")]]
    manifest = {"config": cfg, "sha256": {p.name: sha256_file(p) for p in sources},
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows, failures = [], []
    started = time.monotonic()
    settings = itertools.product(cfg["sizes"], cfg["densities"], cfg["noise_sds"], cfg["scenarios"])
    for size, density, noise, scenario in settings:
        for rep in range(cfg["replicates"]):
            seed = cfg["seed"]+rep
            info = {"participants": size[0], "stimuli": size[1], "density": density, "noise_sd": noise,
                        "scenario": scenario, "replicate": rep, "seed": seed}
            try:
                table, truth = simulate_oracle(seed=seed, participants=size[0], stimuli=size[1],
                    density=density, noise_sd=noise, scenario=scenario,
                    participant_doses=cfg["participant_doses"], stimulus_doses=cfg["stimulus_doses"])
                point, draws, grid, support = paired_primary_draws(table,
                    iterations=cfg["bootstrap_iterations"], seed=seed+100000)
                if len(draws) < .95*cfg["bootstrap_iterations"]:
                    raise ValueError("More than 5% of crossed draws are degenerate")
                candidates = intervals(point, draws, min(support["n_participants"], support["n_stimuli"])-1)
                truth = truth.loc[truth.contrast.eq("eeg_added_matched_calibration")].set_index(
                    ["participant_dose", "stimulus_dose"])
                for method, (lo, hi) in candidates.items():
                    for j, (p, s) in enumerate(grid):
                        target = float(truth.loc[(p, s), "true_delta"])
                        rows.append(dict(**info, **support, method=method, participant_dose=p,
                            stimulus_dose=s, truth=target, estimate=float(point[j]),
                            bootstrap_mean=float(draws[:, j].mean()),
                            bootstrap_sd=float(draws[:, j].std(ddof=1)),
                            ci_low=float(lo[j]), ci_high=float(hi[j]),
                            covered=bool(lo[j]-1e-12 <= target <= hi[j]+1e-12),
                            miss_below=bool(hi[j] < target-1e-12),
                            miss_above=bool(lo[j] > target+1e-12),
                            positive=bool(lo[j] > 1e-12)))
            except ValueError as exc:
                failures.append(dict(**info, error=str(exc)))
        print(size, density, noise, scenario, "done", flush=True)
        pd.DataFrame(rows).to_csv(args.output/"replicates.csv", index=False)
        (args.output/"failures.json").write_text(json.dumps(failures, indent=2)+"\n")
    table = pd.DataFrame(rows)
    keys = ["participants", "stimuli", "density", "noise_sd", "scenario", "method",
            "participant_dose", "stimulus_dose"]
    summary = []
    for identity, group in table.groupby(keys):
        lo, hi = wilson_interval(int(group.covered.sum()), len(group))
        summary.append(dict(zip(keys, identity), replicates=len(group),
            coverage=group.covered.mean(), coverage_mc_low=lo, coverage_mc_high=hi,
            truth=group.truth.iloc[0], bias=(group.estimate-group.truth).mean(),
            empirical_sd=group.estimate.std(ddof=1), mean_bootstrap_sd=group.bootstrap_sd.mean(),
            bootstrap_bias=(group.bootstrap_mean-group.estimate).mean(),
            miss_below=group.miss_below.mean(), miss_above=group.miss_above.mean(),
            positive_rate=group.positive.mean(), mean_width=(group.ci_high-group.ci_low).mean()))
    pd.DataFrame(summary).to_csv(args.output/"summary.csv", index=False)
    (args.output/"completion.json").write_text(json.dumps({
        "status": "executed_not_automatically_validated", "failures": len(failures),
        "elapsed_seconds": time.monotonic()-started,
        "predictions_sha256": sha256_file(args.output/"replicates.csv")}, indent=2)+"\n")


if __name__ == "__main__":
    main()
