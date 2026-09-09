from pathlib import Path

import pytest

from openaffect_eeg.submission import (
    SubmissionError,
    audit_submission_sources,
    parse_latex_label_page,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_latex_label_page_reads_main_content_boundary(tmp_path: Path) -> None:
    aux = _write(
        tmp_path / "main.aux",
        "\\newlabel{main-content-end}{{9}{8}{Conclusion}{section.9}{}}\n",
    )

    assert parse_latex_label_page(aux, "main-content-end") == 8


def test_parse_latex_label_page_rejects_missing_boundary(tmp_path: Path) -> None:
    aux = _write(tmp_path / "main.aux", "\\relax\n")

    with pytest.raises(SubmissionError, match="main-content-end"):
        parse_latex_label_page(aux, "main-content-end")


def test_submission_source_audit_accepts_anonymous_eandd_draft(tmp_path: Path) -> None:
    main = _write(
        tmp_path / "main.tex",
        (
            "\\usepackage[eandd]{neurips_2026}\n"
            "\\author{Anonymous Authors}\n"
            "\\input{supplement}\n\\input{checklist}\n"
        ),
    )
    supplement = _write(tmp_path / "supplement.tex", "Appendix text.\n")
    checklist = _write(
        tmp_path / "checklist.tex",
        "Question. Answer: \\answerYes{} Justification: Complete.\n",
    )

    report = audit_submission_sources(main, supplement, checklist)

    assert report["status"] == "pass"
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("main_extra", "checklist_text", "expected"),
    [
        ("\\usepackage[eandd, final]{neurips_2026}\n", "Complete", "final"),
        ("", "\\answerTODO{}", "TODO"),
        ("Author author" + "@example.org", "Complete", "email"),
    ],
)
def test_submission_source_audit_rejects_release_gate_violations(
    tmp_path: Path,
    main_extra: str,
    checklist_text: str,
    expected: str,
) -> None:
    main = _write(
        tmp_path / "main.tex",
        (
            "\\usepackage[eandd]{neurips_2026}\n"
            "\\author{Anonymous Authors}\n"
            f"{main_extra}"
            "\\input{supplement}\n\\input{checklist}\n"
        ),
    )
    supplement = _write(tmp_path / "supplement.tex", "Appendix text.\n")
    checklist = _write(tmp_path / "checklist.tex", checklist_text)

    report = audit_submission_sources(main, supplement, checklist)

    assert report["status"] == "fail"
    assert any(expected.lower() in error.lower() for error in report["errors"])
