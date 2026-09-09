"""Source-specific channel/unit validation before ds006850 EEG feature extraction."""
from __future__ import annotations

import configparser
import csv
from pathlib import Path

import numpy as np

from openaffect_eeg.labram import LABRAM_STANDARD_1020


def physical_header(path):
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text[text.index("[Common Infos]"):])
    common, binary = parser["Common Infos"], parser["Binary Infos"]
    if (common["DataFormat"] != "BINARY" or common["DataOrientation"] != "MULTIPLEXED"
            or binary["BinaryFormat"] != "IEEE_FLOAT_32"
            or binary.get("UseBigEndianOrder", "NO").upper() != "NO"):
        raise ValueError("Unsupported source binary geometry")
    data = (path.parent/common["DataFile"]).resolve()
    if data.parent != path.parent:
        raise ValueError("Data file must stay in the header directory")
    names, scales = [], []
    unit_scale = {"microvolts": 1e-6, "uv": 1e-6, "v": 1.0}
    for index in range(int(common["NumberOfChannels"])):
        fields = next(csv.reader([parser["Channel Infos"][f"Ch{index+1}"]]))
        if len(fields) != 4 or fields[3].lower() not in unit_scale:
            raise ValueError("Unverified physical unit in source header")
        names.append(fields[0].removeprefix("BrainVision RDA_").upper())
        scales.append(float(fields[2])*unit_scale[fields[3].lower()])
    if len(names) != len(set(names)) or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("Invalid channel names or resolution")
    picks = [i for i, name in enumerate(names) if name not in {"ECG", "GSR_MR_50_XX"}]
    if any(names[i] not in LABRAM_STANDARD_1020 for i in picks):
        raise ValueError("Unmapped source EEG channel")
    return data, names, np.asarray(scales), picks


def conversion_factors(stored, returned, physical_scales):
    """Determine and verify reader-to-volt correction using actual source bytes."""
    stored, returned = np.asarray(stored, float), np.asarray(returned, float)
    expected = stored*np.asarray(physical_scales)[:, None]
    factors = []
    for wanted, actual in zip(expected, returned, strict=True):
        mask = (actual != 0) & np.isfinite(actual) & np.isfinite(wanted)
        if mask.sum() < 2:
            raise ValueError("Insufficient nonzero source samples to verify units")
        factor = float(np.median(wanted[mask]/actual[mask]))
        if not np.isclose(factor, 1, rtol=1e-7, atol=0) and not np.isclose(factor, 1e-6, rtol=1e-7, atol=0):
            raise ValueError("Unexpected reader unit conversion")
        np.testing.assert_allclose(actual*factor, wanted, rtol=1e-7, atol=1e-15)
        factors.append(factor)
    return np.asarray(factors)


def open_urban_eeg(path):
    """Open one recording lazily and return its verified volts correction."""
    import mne

    data, names, scales, picks = physical_header(path)
    raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    if len(raw.ch_names) != len(names):
        raise ValueError("Reader/header channel count mismatch")
    complete_frames = data.stat().st_size // (4 * len(names))
    stored = np.memmap(
        data,
        dtype="<f4",
        mode="r",
        shape=(complete_frames * len(names),),
    ).reshape(-1, len(names))[:128].T
    returned = raw.get_data(start=0, stop=128)
    factors = conversion_factors(stored[picks], returned[picks], scales[picks])
    report = {"input_channels": len(names), "eeg_channels": len(picks),
        "excluded_channels": [name for i, name in enumerate(names) if i not in picks],
        "reader_to_volts_factors": sorted(set(factors.tolist())), "sfreq": float(raw.info["sfreq"]),
        "boundary": "EEG only; source ECG/GSR labels are not trusted as EEG types"}
    raw.pick(picks)
    raw.set_channel_types(
        {name: "eeg" for name in raw.ch_names}, on_unit_change="ignore"
    )
    raw.rename_channels(dict(zip(raw.ch_names, [names[i] for i in picks], strict=True)))
    return raw, factors, report


def read_urban_segment(raw, factors, *, start: int, stop: int):
    """Read an EEG-only interval and convert source-reader values to volts."""
    if start < 0 or stop <= start or stop > raw.n_times:
        raise ValueError(f"EEG segment [{start}, {stop}) exceeds {raw.n_times} samples")
    values = raw.get_data(start=start, stop=stop)
    correction = np.asarray(factors, dtype=float)
    if values.shape[0] != len(correction):
        raise ValueError("EEG segment channels do not match the verified correction")
    result = values * correction[:, None]
    if not np.isfinite(result).all():
        raise ValueError("EEG segment contains non-finite physical values")
    return result


def nominal_sampling_frequency(measured: float, *, expected: float = 500.0, tolerance_ppm: float = 50.0):
    """Validate source clock drift before using the documented nominal rate."""
    if not np.isfinite(measured) or measured <= 0 or expected <= 0 or tolerance_ppm <= 0:
        raise ValueError("Sampling frequencies and tolerance must be positive")
    drift_ppm = abs(float(measured)-float(expected))/float(expected)*1_000_000
    if drift_ppm > tolerance_ppm:
        raise ValueError(
            f"Measured sampling frequency {measured:g} differs from nominal "
            f"{expected:g} by {drift_ppm:.3f} ppm"
        )
    return float(expected), float(drift_ppm)


def trial_sample_bounds(row, *, sampling_frequency: float, available_samples: int):
    """Use the event's exact one-based sample and keep the epoch before rating."""
    start = int(row.eeg_sample_1based) - 1
    count = round(float(row.eeg_duration_s) * sampling_frequency)
    stop = start + count
    if start < 0 or count <= 0 or stop > available_samples:
        raise ValueError(
            f"Trial {row.trial_uid} bounds [{start}, {stop}) exceed "
            f"{available_samples} samples"
        )
    onset_start = round(float(row.eeg_onset_s) * sampling_frequency)
    return start, stop, abs(start-onset_start)


def read_urban_eeg(path, *, preload=False):
    """Use MNE I/O with audited EEG-only picks and physical-volt correction."""
    raw, factors, report = open_urban_eeg(path)
    if not preload:
        raw.close()
        return None, report
    expected = raw.get_data(start=0, stop=128) * factors[:, None]
    raw.load_data()
    raw.apply_function(lambda values: values*factors[:, None], channel_wise=False,
                       picks="all", n_jobs=1, dtype=np.float64, verbose="ERROR")
    np.testing.assert_allclose(
        raw.get_data(start=0, stop=128), expected, rtol=1e-7, atol=1e-15
    )
    return raw, report
