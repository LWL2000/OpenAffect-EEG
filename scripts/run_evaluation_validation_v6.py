"""Run the frozen v6 method-validation experiments; all results are synthetic."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.evaluation_validation import simulate_oracle, simulate_trials, summarize_oracle
from openaffect_eeg.exposure_statistics import analyze_predictions
from openaffect_eeg.identity_exposure import compile_exposure_cell, evaluate_exposure_cell
from openaffect_eeg.splits import build_double_holdout_split


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["oracle", "pipeline"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).parents[1]/"src/openaffect_eeg/evaluation_validation.py",
               Path(__file__).parents[1]/"src/openaffect_eeg/identity_exposure.py",
               Path(__file__).parents[1]/"src/openaffect_eeg/splits.py",
               Path(__file__).parents[1]/"src/openaffect_eeg/baselines.py",
               Path(__file__).parents[1]/"src/openaffect_eeg/exposure_statistics.py"]
    fingerprint = {"config_sha256": sha256_file(args.config), "config": config,
        "source_sha256": {p.name: sha256_file(p) for p in sources}, "mode": args.mode,
        "smoke": args.smoke, "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__}
    manifest = output/"run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("Run fingerprint changed; retain results and use a new output directory")
    write_json(manifest, fingerprint)
    if (output/"completion.json").exists():
        print("Already complete; retaining the existing run")
        return
    replicates = config["replicates" if args.mode == "oracle" else "pipeline_replicates"]
    replicates = min(2, replicates) if args.smoke else replicates
    draws = 100 if args.smoke else config["bootstrap_draws"]
    p_values, s_values = config["participant_doses"], config["stimulus_doses"]
    noises = config["noise_sd"] if args.mode == "oracle" else [config["pipeline_noise_sd"]]
    started = time.monotonic()
    rows, failures = [], []
    directory = output/"replicates"
    directory.mkdir(exist_ok=True)
    settings = list(itertools.product(config["scenarios"], config["densities"], noises))
    for setting, (scenario, density, noise) in enumerate(settings):
        for replicate in range(replicates):
            # Independent datasets across repetitions; paired scenarios share draws.
            seed = config["seed"] + replicate*100003
            label = f"setting-{setting:02d}_replicate-{replicate:03d}"
            path, failed_path = directory/f"{label}.csv", directory/f"{label}.failure.json"
            context = dict(scenario=scenario, density=density, noise_sd=noise, replicate=replicate)
            if failed_path.exists():
                failures.append(json.loads(failed_path.read_text()))
                continue
            if path.exists():
                rows.append(pd.read_csv(path))
                continue
            try:
                if args.mode == "oracle":
                    predictions, exact = simulate_oracle(seed=seed, participants=config["participants"],
                        stimuli=config["stimuli"], density=density, noise_sd=noise, scenario=scenario,
                        participant_doses=p_values, stimulus_doses=s_values)
                    _, contrasts, _, report = analyze_predictions(predictions, iterations=draws, seed=seed+41)
                    result = contrasts.merge(exact, on=["participant_dose", "stimulus_dose", "contrast"], validate="1:1")
                    result["n_test_trials"] = report["supports"][0]["n_trials"]
                else:
                    trials, features = simulate_trials(seed=seed, participants=config["participants"],
                        stimuli=config["stimuli"], density=density, noise_sd=noise, scenario=scenario)
                    strict = build_double_holdout_split(trials, first_group="subject_uid",
                        second_group="stimulus_uid", seed=seed, validation_fraction=.15, test_fraction=.3)
                    collected, audits = [], []
                    for p, s in itertools.product(p_values, s_values):
                        assignment, audit = compile_exposure_cell(trials, strict,
                            participant_dose=p, stimulus_dose=s, maximum_participant_dose=max(p_values),
                            maximum_stimulus_dose=max(s_values), seed=seed)
                        _, predictions = evaluate_exposure_cell(trials, assignment, features,
                            alphas=tuple(config["ridge_alphas"]))
                        predictions["representation"] = scenario
                        collected.append(predictions)
                        audits.append(audit)
                    if len({a["fixed_test_support_sha256"] for a in audits}) != 1:
                        raise ValueError("Production simulation lost fixed support")
                    _, result, _, report = analyze_predictions(pd.concat(collected), iterations=draws, seed=seed+41)
                    result["n_test_trials"] = report["supports"][0]["n_trials"]
                    result["n_test_participants"] = report["supports"][0]["n_participants"]
                    result["n_test_stimuli"] = report["supports"][0]["n_stimuli_or_conditions"]
                    result["true_delta"] = np.nan
                for key, value in context.items():
                    result[key] = value
                temporary = path.with_suffix(".pending.csv")
                result.to_csv(temporary, index=False)
                os.replace(temporary, path)
                rows.append(result)
            except Exception as exc:
                failure = {**context, "error_type": type(exc).__name__, "error": str(exc)}
                write_json(failed_path, failure)
                failures.append(failure)
            if (replicate+1) % 10 == 0 or replicate+1 == replicates:
                print(label, "completed", "failures", len(failures), "seconds", round(time.monotonic()-started, 1), flush=True)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.to_csv(output/"replicate_results.csv", index=False)
    write_json(output/"failures.json", failures)
    if args.mode == "oracle" and len(combined):
        summary = summarize_oracle(combined)
        summary.to_csv(output/"operating_characteristics.csv", index=False)
        grouped = combined.groupby(["scenario", "density", "noise_sd", "contrast", "replicate"])
        simultaneous = grouped.apply(lambda g: bool(((g.ccc_simultaneous_low <= g.true_delta+1e-12)
            & (g.true_delta <= g.ccc_simultaneous_high+1e-12)).all()), include_groups=False)
        simultaneous.rename("all_cells_covered").reset_index().to_csv(output/"simultaneous_coverage.csv", index=False)
    write_json(output/"completion.json", {"status": "completed_with_failures" if failures else "completed",
        "settings": len(settings), "requested_replicates_per_setting": replicates,
        "completed_replicates": len(rows), "failed_replicates": len(failures),
        "elapsed_seconds_this_invocation": time.monotonic()-started,
        "run_manifest_sha256": sha256_file(manifest),
        "result_sha256": sha256_file(output/"replicate_results.csv"),
        "interpretation": "Execution complete is not statistical validation passed; inspect coverage and failures."})


if __name__ == "__main__":
    main()
