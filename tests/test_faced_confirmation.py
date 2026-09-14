from __future__ import annotations

import numpy as np
import pytest

from openaffect_eeg.faced_confirmation import (
    CHANNELS,
    LEGACY_CHANNELS,
    MODERN_CHANNELS_WITH_EOG,
    _event_windows,
    _preprocess_window,
    _read_targets,
    canonical_eeg_indices,
    parse_bdf_header,
)


def synthetic_bdf_header(*, records: int = 10, sampling_hz: int = 250) -> bytes:
    signals = len(LEGACY_CHANNELS)
    header_bytes = 256 + 256 * signals
    fixed = bytearray(b" " * 256)
    fixed[0:8] = b"\xffBIOSEMI"
    fixed[184:192] = f"{header_bytes:<8}".encode()
    fixed[236:244] = f"{records:<8}".encode()
    fixed[244:252] = f"{1:<8}".encode()
    fixed[252:256] = f"{signals:<4}".encode()
    fields = bytearray()
    fields.extend(b"".join(f"{name:<16}".encode() for name in LEGACY_CHANNELS))
    fields.extend(b" " * (signals * (80 + 8 + 8 + 8 + 8 + 8 + 80)))
    fields.extend(b"".join(f"{sampling_hz:<8}".encode() for _ in LEGACY_CHANNELS))
    fields.extend(b" " * (signals * 32))
    return bytes(fixed + fields)


def test_parse_bdf_header_validates_expected_size_and_channels() -> None:
    block = synthetic_bdf_header()
    expected_size = len(block) + 10 * len(LEGACY_CHANNELS) * 250 * 3
    parsed = parse_bdf_header(block, file_size=expected_size)
    assert parsed["labels"] == list(LEGACY_CHANNELS)
    assert set(parsed["sampling_rates"]) == {250.0}
    assert parsed["expected_file_size"] == expected_size


def test_both_source_montages_map_to_the_same_30_scalp_channels() -> None:
    legacy_indices, legacy_name = canonical_eeg_indices(LEGACY_CHANNELS)
    modern_indices, modern_name = canonical_eeg_indices(MODERN_CHANNELS_WITH_EOG)
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    legacy_selected = [aliases.get(LEGACY_CHANNELS[index], LEGACY_CHANNELS[index]) for index in legacy_indices]
    modern_selected = [MODERN_CHANNELS_WITH_EOG[index] for index in modern_indices]
    assert legacy_selected == list(CHANNELS)
    assert modern_selected == list(CHANNELS)
    assert legacy_name != modern_name


def test_event_windows_use_last_30_seconds_without_reading_ratings() -> None:
    header = "onset\tduration\tvideo_index\tValence\tArousal\n"
    rows = [f"{index * 40}\t35\t{index}\tSECRET\tSECRET" for index in range(1, 29)]
    windows = _event_windows(header + "\n".join(rows) + "\n")
    assert windows[0] == ("1", 45.0, 75.0)
    assert windows[-1] == ("28", 1125.0, 1155.0)


def test_targets_are_unsealed_only_within_the_frozen_range(tmp_path) -> None:
    path = tmp_path / "events.tsv"
    header = "onset\tduration\tvideo_index\tValence\tArousal\n"
    rows = [f"{index * 40}\t35\t{index}\t3.5\t7" for index in range(1, 29)]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    targets = _read_targets(path)
    assert targets["1"] == (0.0, 1.0)
    path.write_text((header + "\n".join(rows) + "\n").replace("3.5\t7", "3.5\t8", 1))
    with pytest.raises(ValueError, match="out-of-range"):
        _read_targets(path)


def test_preprocess_window_returns_frozen_shape_and_units() -> None:
    sampling_hz = 250.0
    time = np.arange(round(30 * sampling_hz)) / sampling_hz
    base = np.sin(2 * np.pi * 10 * time) * 20e-6
    data = np.stack([(index + 1) * base for index in range(len(CHANNELS))])

    class FakeRaw:
        def __init__(self) -> None:
            self.info = {"sfreq": sampling_hz}

        def get_data(self, *, picks, start, stop):
            assert picks == list(range(len(CHANNELS)))
            return data[:, start:stop]

    tensor = _preprocess_window(
        FakeRaw(), list(range(len(CHANNELS))), start_seconds=0.0, stop_seconds=30.0
    )
    assert tensor.shape == (len(CHANNELS), 3000)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
