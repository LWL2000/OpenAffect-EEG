"""Build genuine one-target pre-rating Urban EEG tensors for the v9 check."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.urban_brainvision import (
    nominal_sampling_frequency, open_urban_eeg, read_urban_segment, trial_sample_bounds,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    root = args.data_root
    manifest = root / "derived/urban_confirmation_v7_metadata4/valence_trials.tsv.gz"
    out = root / "derived/final_robustness_v9_inputs"
    out.mkdir(parents=True, exist_ok=True)
    archive, uids_path, meta_path = [out / name for name in ("urban_100hz.npy", "urban_100hz_trial_uids.npy", "urban_100hz.json")]
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["manifest_sha256"] != sha256_file(manifest) or metadata["array_sha256"] != sha256_file(archive) or metadata["uids_sha256"] != sha256_file(uids_path):
            raise ValueError("Urban tensor resume integrity mismatch")
        print("Validated existing Urban tensors", flush=True)
        return
    table = pd.read_csv(manifest, sep="\t").sort_values("trial_uid").reset_index(drop=True)
    temporary = out / "urban_100hz.incomplete.npy"
    tensors = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(table), 64, 300))
    covered = np.zeros(len(table), dtype=bool)
    channel_order = None
    for number, (relative, rows) in enumerate(table.groupby("eeg_path", sort=True), 1):
        raw, factors, _ = open_urban_eeg(root / "raw/openneuro/ds006850" / relative)
        try:
            nominal, _ = nominal_sampling_frequency(float(raw.info["sfreq"]))
            if nominal != 500:
                raise ValueError("Unexpected source sampling frequency")
            if channel_order is None:
                channel_order = list(raw.ch_names)
            if list(raw.ch_names) != channel_order:
                raise ValueError("Inconsistent channel mapping")
            for row in rows.itertuples():
                start, stop, _ = trial_sample_bounds(row, sampling_frequency=nominal, available_samples=raw.n_times)
                values = read_urban_segment(raw, factors, start=start, stop=stop)
                values = values - values.mean(axis=0, keepdims=True)
                sample = resample_poly(values, up=1, down=5, axis=1)
                if sample.shape != (64, 300) or not np.isfinite(sample).all():
                    raise ValueError("Invalid pre-rating EEG epoch")
                tensors[row.Index] = sample
                covered[row.Index] = True
        finally:
            raw.close()
        if number % 20 == 0:
            print("Urban recordings prepared", number, flush=True)
    if not covered.all():
        raise ValueError("Incomplete tensor coverage")
    tensors.flush()
    del tensors
    temporary.replace(archive)
    np.save(uids_path, table.trial_uid.to_numpy(str), allow_pickle=False)
    metadata = dict(manifest_sha256=sha256_file(manifest), array_sha256=sha256_file(archive),
                    uids_sha256=sha256_file(uids_path), shape=[len(table), 64, 300],
                    units="physical volts", reference="per-sample common average", sampling_hz=100,
                    window="three seconds before rating onset", resampling="scipy.signal.resample_poly 1/5",
                    channel_order=channel_order, target="valence only, no fabricated second label")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("Urban tensors verified", metadata["shape"], flush=True)


if __name__ == "__main__":
    main()
