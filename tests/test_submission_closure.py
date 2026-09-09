import json
from pathlib import Path

from openaffect_eeg.submission_closure import build_submission_closure


ROOT = Path(__file__).parents[1]


def test_rule_catalog_has_three_state_examples_and_unique_rules() -> None:
    catalog = json.loads((ROOT / "configs/audit_rule_catalog.json").read_text())
    rules = catalog["rules"]
    assert len({rule["id"] for rule in rules}) == len(rules)
    for rule in rules:
        assert rule["evaluator_checks"]
        assert rule["basis"]
        assert rule["allow_example"]
        assert rule["block_example"]
        assert rule["unverifiable_example"]
        assert rule["unsupported_inference"]


def test_submission_closure_rebuilds_from_frozen_sources(tmp_path: Path) -> None:
    manifest = build_submission_closure(ROOT, tmp_path)
    validation = json.loads((tmp_path / "evaluation_method_validation.json").read_text())
    repair = json.loads((tmp_path / "external_claim_repair.json").read_text())

    assert manifest["status"] == "generated_from_hash_verified_frozen_sources"
    assert validation["v6"]["trial_signal_positive_detection_rate_range"] == [0.85, 1.0]
    assert validation["v6"]["trial_signal_pointwise_coverage_range"] == [0.92, 1.0]
    assert validation["v6"]["independent_noise_positive_detection_rate_range"] == [0.0, 0.0]
    assert validation["v7"]["target_32_by_24_coverage"]["percentile"] == 0.9333333333333333
    assert validation["v7"]["small_8_by_6_min_coverage"]["basic"] == 0.82
    assert repair["strict_claim"]["status"] == "block"
    assert repair["strict_claim"]["folds"] == 5
    assert repair["scoped_claim"]["status"] == "allow"
    assert repair["scoped_claim"]["folds"] == 5


def test_submission_closure_outputs_use_platform_independent_newlines(
    tmp_path: Path,
) -> None:
    build_submission_closure(ROOT, tmp_path)

    for name in (
        "evaluation_method_validation.json",
        "external_claim_repair.json",
        "manifest.json",
    ):
        assert b"\r\n" not in (tmp_path / name).read_bytes()
