"""Extract strict image-epoch band power from audited ds006850 BrainVision data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.features import BANDS, global_bandpower_summary, log_bandpower
from openaffect_eeg.urban_brainvision import (
    nominal_sampling_frequency,
    open_urban_eeg,
    read_urban_segment,
    trial_sample_bounds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    trials = (
        pd.read_csv(args.manifest, sep="\t")
        .sort_values("trial_uid")
        .reset_index(drop=True)
    )
    feature_rows: list[np.ndarray | None] = [None] * len(trials)
    global_rows: list[np.ndarray | None] = [None] * len(trials)
    canonical_channels = None
    groups = trials.groupby("eeg_path", sort=True)
    for recording_index, (relative, rows) in enumerate(groups, start=1):
        raw, factors, _ = open_urban_eeg(args.dataset_root / str(relative))
        try:
            nominal, _ = nominal_sampling_frequency(float(raw.info["sfreq"]))
            channels = list(raw.ch_names)
            if canonical_channels is None:
                canonical_channels = channels
            elif channels != canonical_channels:
                raise ValueError(f"EEG channel order differs in {relative}")
            for row in rows.itertuples(index=True):
                start, stop, _ = trial_sample_bounds(
                    row, sampling_frequency=nominal, available_samples=raw.n_times
                )
                values = read_urban_segment(raw, factors, start=start, stop=stop)
                values -= values.mean(axis=0, keepdims=True)
                channel = log_bandpower(values, nominal)
                feature_rows[row.Index] = channel.astype(np.float32)
                global_rows[row.Index] = global_bandpower_summary(channel).astype(
                    np.float32
                )
        finally:
            raw.close()
        if (
            recording_index == 1
            or recording_index % 10 == 0
            or recording_index == groups.ngroups
        ):
            print(
                f"Band power recordings {recording_index}/{groups.ngroups}", flush=True
            )

    if any(row is None for row in feature_rows + global_rows):
        raise ValueError("Band-power extraction did not cover every trial")
    channel_array = np.stack(feature_rows)
    global_array = np.stack(global_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        trial_uid=trials["trial_uid"].to_numpy(str),
        channel_log_bandpower=channel_array,
        global_log_bandpower=global_array,
        channel_names=np.asarray(canonical_channels),
        band_names=np.asarray([name for name, _, _ in BANDS]),
    )
    temporary.replace(args.output)
    metadata = {
        "dataset_id": "ds006850",
        "feature_shape": list(channel_array.shape),
        "global_feature_shape": list(global_array.shape),
        "manifest_sha256": sha256_file(args.manifest),
        "output_sha256": sha256_file(args.output),
        "bands_hz": {name: [low, high] for name, low, high in BANDS},
        "sampling_frequency_hz": 500.0,
        "channel_reference": "per-sample common average over 64 audited EEG channels",
        "window": "three-second image epoch ending before scale presentation",
        "welch_window_s": 2.0,
        "source_units": "verified physical volts",
        "trial_count": len(trials),
        "channel_order": canonical_channels,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key != "channel_order"}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
