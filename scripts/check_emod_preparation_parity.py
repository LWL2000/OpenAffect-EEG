"""Compare prepared training windows with the pinned official preprocessing."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from prepare_emod_seedv_v3 import literal_assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing parity report")
    sys.path.insert(0, str(args.source))
    preprocess = importlib.import_module("data_preprocess.main_preprocessing").Preprocessing
    channel_module = importlib.import_module("data_preprocess.channel_idx")
    official_path = args.source/"data_preprocess/SEEDV_preprocess.py"
    constants = literal_assignments(official_path, ["trials_of_sessions", "labels_of_sessions", "useless_ch"])
    manifest_path = args.prepared/"prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["channel_ids"] != channel_module.seed_channel_ids or manifest["channel_names"] != channel_module.seed_channels:
        raise ValueError("Channel metadata differs from the official vocabulary")
    checks, subset = [], {0, 23, 47}
    # Include every source-reader fallback and the largest-amplitude source file.
    subset.update(i for i, r in enumerate(manifest["records"]) if r["data_format"] != "auto")
    subset.add(int(np.argmax([r["microvolt_std"] for r in manifest["records"]])))
    for i, record in enumerate(manifest["records"]):
        table = pd.read_csv(args.prepared/record["table_file"], sep="\t")
        labels = constants["labels_of_sessions"][str(record["session"])]
        train = table.loc[table["split"].eq("train")]
        trial_indices = train.trial_uid.str.rsplit(":", n=1).str[-1].astype(int)
        if not (trial_indices < 5).all() or not np.array_equal(train.label, [labels[k] for k in trial_indices]):
            raise ValueError("Training labels differ from official trial constants")
        if i not in subset:
            continue
        paths = [p for p in args.raw.glob(f"{record['subject']}_{record['session']}_*.cnt") if "repaired" not in p.name.lower()]
        if len(paths) != 1 or sha256_file(paths[0]) != record["fingerprint"]["raw_sha256"]:
            raise ValueError("Original raw identity/hash mismatch")
        raw = mne.io.read_raw_cnt(paths[0], preload=True,
            data_format="int32" if record["data_format"] == "int32_fallback" else "auto", verbose="ERROR")
        raw.drop_channels(constants["useless_ch"])
        if [x.upper() for x in raw.ch_names] != [x.upper() for x in manifest["channel_names"]]:
            raise ValueError("Official source channel order differs from our prepared order")
        raw.resample(200, n_jobs=1, verbose="ERROR")
        pipeline = preprocess(raw)
        pipeline.band_pass_filter(.3, 49)
        pipeline.average_ref()
        signal = pipeline.raw.get_data(units="uV")
        saved = np.load(args.prepared/record["tensor_file"], mmap_mode="r", allow_pickle=False)
        rows = train.iloc[np.linspace(0, len(train)-1, 20, dtype=int)]
        differences = []
        for idx, row in rows.iterrows():
            trial = int(row.trial_uid.rsplit(":", 1)[-1])
            second = constants["trials_of_sessions"][str(record["session"])]["start"][trial] + int(row.window_offset_seconds)
            reference = signal[:, second*200:(second+1)*200].astype(np.float32)
            differences.append(float(np.abs(reference-saved[idx]).max()))
        maximum = max(differences)
        if maximum > 1e-5:
            raise ValueError(f"Prepared/reference training-window mismatch: {maximum}")
        checks.append({"record_index": i, "raw_sha256": record["fingerprint"]["raw_sha256"],
                       "windows": len(rows), "max_abs_microvolt_difference": maximum,
                       "original_channel_order_matches": True, "data_format": record["data_format"]})
        print("parity", i, maximum, flush=True)
        del raw, pipeline, signal, saved
    report = {"status": "pass", "checked_training_label_files": len(manifest["records"]),
        "prepared_manifest_sha256": sha256_file(manifest_path), "script_sha256": sha256_file(Path(__file__)),
        "source_sha256": {p.name: sha256_file(p) for p in [official_path, args.source/"data_preprocess/main_preprocessing.py"]},
        "raw_reprocessing_checks": checks,
        "boundary": "Training windows only; selected raw-file parity is not exhaustive independent ground-truth validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
