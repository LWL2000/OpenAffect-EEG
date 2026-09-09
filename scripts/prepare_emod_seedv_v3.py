"""Authorized SEED-V source-task preparation using official EMOD operations.

Private raw EEG stays on the server. This within-participant protocol checks
task compatibility; it is not an independent-participant confirmation study.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file


def literal_assignments(path, names):
    result = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                result[node.targets[0].id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError("Required official preprocessing constants not found")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    channels = importlib.import_module("data_preprocess.channel_idx")
    official = args.source / "data_preprocess/SEEDV_preprocess.py"
    constants = literal_assignments(official, ["trials_of_sessions", "labels_of_sessions"])
    times, labels = constants["trials_of_sessions"], constants["labels_of_sessions"]
    source_hashes = {str(p.relative_to(args.source)): sha256_file(p) for p in
        (official, args.source/"data_preprocess/main_preprocessing.py", args.source/"data_preprocess/channel_idx.py")}
    files = {}
    for path in sorted(args.raw_root.glob("*.cnt")):
        match = re.match(r"^(\d+)_(\d+)_.*\.cnt$", path.name, flags=re.I)
        if not match:
            raise ValueError(f"Unrecognized CNT filename: {path.name}")
        key = tuple(map(int, match.groups()))
        if "repaired" in path.stem.lower():
            continue
        if key in files:
            raise ValueError(f"Duplicate original CNT file for {key}")
        files[key] = path
    if set(files) != {(i, j) for i in range(1, 17) for j in (1, 2, 3)}:
        raise ValueError("Expected exactly 16 participants by three sessions")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for (subject, session), path in sorted(files.items()):
        stem = f"subject-{subject:02d}_session-{session}"
        tensor_path, table_path, manifest_path = (args.output / f"{stem}{suffix}" for suffix in (".npy", ".tsv", ".json"))
        fingerprint = {"raw_sha256": sha256_file(path), "source_hashes": source_hashes,
                       "script_sha256": sha256_file(Path(__file__))}
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved["fingerprint"] != fingerprint or saved["tensor_sha256"] != sha256_file(tensor_path) or saved["table_sha256"] != sha256_file(table_path):
                raise ValueError("Prepared-data resume hash mismatch")
            records.append(saved)
            continue
        try:
            raw = mne.io.read_raw_cnt(path, preload=True, verbose="ERROR")
            data_format = "auto"
        except (ValueError, RuntimeError, IndexError):
            raw = mne.io.read_raw_cnt(path, preload=True, data_format="int32", verbose="ERROR")
            data_format = "int32_fallback"
        names = {x.upper(): x for x in raw.ch_names}
        raw.pick([names[x.upper()] for x in channels.seed_channels])
        raw.resample(200, n_jobs=1, verbose="ERROR")
        raw.filter(.3, 49, n_jobs=1, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        data = raw.get_data(units="uV")
        rows, windows = [], []
        for trial, (start, end, label) in enumerate(zip(times[str(session)]["start"], times[str(session)]["end"], labels[str(session)], strict=True)):
            x = data[:, start*200:end*200]
            if x.shape != (62, (end-start)*200):
                raise ValueError("CNT duration does not cover the official stimulus interval")
            windows.append(x.reshape(62, -1, 200).transpose(1, 0, 2).astype(np.float32))
            split = "train" if trial < 5 else "validation" if trial < 10 else "test"
            for offset in range(end-start):
                rows.append({"trial_uid": f"SEEDV:{subject}:{session}:{trial}",
                    "subject_uid": f"SEEDV:{subject}", "session_uid": f"SEEDV:{subject}:{session}",
                    "stimulus_position_uid": f"SEEDV:session-{session}:position-{trial}",
                    "window_offset_seconds": offset, "split": split, "label": label})
        values = np.concatenate(windows)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite prepared EEG")
        np.save(tensor_path, values, allow_pickle=False)
        pd.DataFrame(rows).to_csv(table_path, sep="\t", index=False)
        saved = {"subject": subject, "session": session, "fingerprint": fingerprint,
            "tensor_file": tensor_path.name, "table_file": table_path.name,
            "tensor_sha256": sha256_file(tensor_path), "table_sha256": sha256_file(table_path),
            "shape": list(values.shape), "data_format": data_format,
            "microvolt_std": float(values.std()), "microvolt_max_abs": float(np.abs(values).max())}
        manifest_path.write_text(json.dumps(saved, indent=2)+"\n", encoding="utf-8")
        records.append(saved)
        print(stem, values.shape, "verified", flush=True)
        del raw, data, windows, values
    manifest = {"dataset": "authorized_SEEDV", "records": records,
        "channel_ids": channels.seed_channel_ids, "channel_names": channels.seed_channels,
        "source_hashes": source_hashes, "units": "uV", "sample_rate": 200,
        "preprocessing": "continuous resample -> 0.3-49 Hz MNE FIR -> average reference -> official 1-s windows",
        "split": "official EMOD first/middle/last five trials within every participant and session",
        "boundary": "Native classification task check, not unseen-participant generalization; continuous offline preprocessing; one repaired duplicate excluded"}
    (args.output / "prepared_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
