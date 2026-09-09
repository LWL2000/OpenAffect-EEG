from pathlib import Path

from openaffect_eeg.manuscript_audit import audit_manuscript


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_manuscript_audit_accepts_resolved_citations_and_claims(tmp_path: Path) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        "Result 0.1234 [@source2024]. <!-- claims: C01 -->\n",
    )
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{source2024,\n  title={Source},\n  year={2024}\n}\n",
    )
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\nsource2024\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "pass"
    assert report["citation_count"] == 1
    assert report["claim_reference_count"] == 1
    assert report["errors"] == []


def test_manuscript_audit_reports_missing_evidence_and_sensitive_text(
    tmp_path: Path,
) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        (
            "Unsupported [@missing2026]. <!-- claims: C99 -->\n"
            "Contact author" + "@example.org and read /" + "home" + "/alice/private.\n"
        ),
    )
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{present2024, title={Present}, year={2024}}\n",
    )
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\npresent2024\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    joined = "\n".join(report["errors"])
    assert "missing2026" in joined
    assert "C99" in joined
    assert "email address" in joined
    assert "local home path" in joined


def test_manuscript_audit_requires_citation_audit_coverage(tmp_path: Path) -> None:
    manuscript = _write(tmp_path / "paper.md", "Evidence [@source2024].\n")
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{source2024, title={Source}, year={2024}}\n",
    )
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\nother2024\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    assert any("citation audit" in error for error in report["errors"])


def test_manuscript_audit_rejects_unmapped_numeric_prose(tmp_path: Path) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        "The reported score was 0.8123, but no evidence mapping was supplied.\n",
    )
    bibliography = _write(tmp_path / "references.bib", "")
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.8123\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    assert report["unmapped_numeric_block_count"] == 1
    assert any("numeric block" in error.lower() for error in report["errors"])


def test_manuscript_audit_accepts_numeric_blocks_with_explicit_sources(
    tmp_path: Path,
) -> None:
    split_config = tmp_path / "configs" / "splits.yaml"
    split_config.parent.mkdir(parents=True)
    split_config.write_text("version: test\n", encoding="utf-8")
    manuscript = _write(
        tmp_path / "paper.md",
        (
            "Result 0.1234. <!-- claims: C01 -->\n\n"
            "The split uses 70/10/20. "
            "<!-- constants-source: configs/splits.yaml -->\n\n"
            "Snapshot version 1.0.3 [@source2024].\n"
        ),
    )
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{source2024, title={Source}, year={2024}}\n",
    )
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\nsource2024\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "pass"
    assert report["numeric_evidence_block_count"] == 3
    assert report["unmapped_numeric_blocks"] == []


def test_manuscript_audit_rejects_missing_local_source_reference(
    tmp_path: Path,
) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        (
            "The crop length is 10 seconds. "
            "<!-- constants-source: configs/missing.yaml -->\n"
        ),
    )
    bibliography = _write(tmp_path / "references.bib", "")
    claim_ledger = _write(tmp_path / "claims.tsv", "claim_id\tdisplay_value\n")
    citation_audit = _write(tmp_path / "citations.tsv", "cite_key\tstatus\n")

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    assert report["missing_source_references"] == ["configs/missing.yaml"]
    assert any("source reference" in error.lower() for error in report["errors"])


def test_manuscript_audit_requires_full_bibliography_inventory_coverage(
    tmp_path: Path,
) -> None:
    manuscript = _write(tmp_path / "paper.md", "No citations in this draft.\n")
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{only_in_bib, title={Source}, year={2024}}\n",
    )
    claim_ledger = _write(tmp_path / "claims.tsv", "claim_id\tdisplay_value\n")
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\nonly_in_audit\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    joined = "\n".join(report["errors"])
    assert "only_in_bib" in joined
    assert "only_in_audit" in joined


def test_manuscript_audit_rejects_numeric_value_that_disagrees_with_claim(
    tmp_path: Path,
) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        "The reported score was 0.9999. <!-- claims: C01 -->\n",
    )
    bibliography = _write(tmp_path / "references.bib", "")
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(tmp_path / "citations.tsv", "cite_key\tstatus\n")

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    assert report["claim_value_mismatches"][0]["value"] == "0.9999"
    assert report["claim_value_mismatches"][0]["claim_ids"] == ["C01"]


def test_manuscript_audit_accepts_value_authorized_by_allowed_wording(
    tmp_path: Path,
) -> None:
    manuscript = _write(
        tmp_path / "paper.md",
        "The baseline had 0.0134 lower MAE. <!-- claims: C01 -->\n",
    )
    bibliography = _write(tmp_path / "references.bib", "")
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        (
            "claim_id\tdisplay_value\tallowed_wording\n"
            "C01\t-0.0134\tThe baseline had 0.0134 lower MAE.\n"
        ),
    )
    citation_audit = _write(tmp_path / "citations.tsv", "cite_key\tstatus\n")

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "pass"
    assert report["claim_value_mismatch_count"] == 0


def test_manuscript_audit_accepts_latex_citations_claims_and_sources(
    tmp_path: Path,
) -> None:
    split_config = tmp_path / "configs" / "splits.yaml"
    split_config.parent.mkdir(parents=True)
    split_config.write_text("version: test\n", encoding="utf-8")
    manuscript = _write(
        tmp_path / "paper.tex",
        (
            "The audited score was \\claimvalue{C01} \\citep{source2024}.\n\n"
            "The split ratio was 70/10/20. "
            "% constants-source: configs/splits.yaml\n"
        ),
    )
    bibliography = _write(
        tmp_path / "references.bib",
        "@article{source2024, title={Source}, year={2024}}\n",
    )
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(
        tmp_path / "citations.tsv",
        "cite_key\tstatus\nsource2024\tverified\n",
    )

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
        source_root=tmp_path,
    )

    assert report["status"] == "pass"
    assert report["citations"] == ["source2024"]
    assert report["claim_references"] == ["C01"]
    assert report["source_references"] == ["configs/splits.yaml"]


def test_manuscript_audit_rejects_unknown_latex_claim_macro(tmp_path: Path) -> None:
    manuscript = _write(
        tmp_path / "paper.tex",
        "The audited score was \\claimvalue{C99}.\n",
    )
    bibliography = _write(tmp_path / "references.bib", "")
    claim_ledger = _write(
        tmp_path / "claims.tsv",
        "claim_id\tdisplay_value\nC01\t0.1234\n",
    )
    citation_audit = _write(tmp_path / "citations.tsv", "cite_key\tstatus\n")

    report = audit_manuscript(
        manuscript,
        bibliography,
        claim_ledger,
        citation_audit,
    )

    assert report["status"] == "fail"
    assert any("C99" in error for error in report["errors"])
