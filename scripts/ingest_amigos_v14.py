#!/usr/bin/env python3
"""Unseal and ingest AMIGOS only after the label-blind QC report is locked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openaffect_eeg.amigos_confirmation import ingest_amigos, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_preprocessed", type=Path)
    parser.add_argument("structural_report", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    structural = json.loads(args.structural_report.read_text(encoding="utf-8"))
    trials, tensors, uids, bandpower, report = ingest_amigos(
        args.data_preprocessed, structural
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    trial_path = args.output_root / "amigos_trials.tsv.gz"
    tensor_path = args.output_root / "amigos_100hz_30s_microvolts.npy"
    uid_path = args.output_root / "amigos_trial_uids.npy"
    feature_path = args.output_root / "amigos_bandpower.npz"
    trials.to_csv(trial_path, sep="\t", index=False)
    np.save(tensor_path, tensors, allow_pickle=False)
    np.save(uid_path, uids, allow_pickle=False)
    np.savez_compressed(feature_path, trial_uid=uids, channel_log_bandpower=bandpower)
    report["structural_report_sha256"] = sha256_file(args.structural_report)
    report["outputs_sha256"] = {
        path.name: sha256_file(path)
        for path in (trial_path, tensor_path, uid_path, feature_path)
    }
    (args.output_root / "ingestion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "participants", "stimuli", "trials")}, sort_keys=True))


if __name__ == "__main__":
    main()

