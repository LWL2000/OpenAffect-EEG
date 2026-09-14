from __future__ import annotations

from openaffect_eeg.faced_confirmation import (
    CHANNELS,
    LEGACY_CHANNELS,
    MODERN_CHANNELS_WITH_EOG,
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
