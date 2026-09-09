"""Deterministic build and release gates for the anonymous NeurIPS paper."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.manuscript_audit import audit_manuscript


class SubmissionError(RuntimeError):
    """Raised when the anonymous submission cannot pass a required gate."""


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LOCAL_PATH = re.compile(
    r"(?<![\w.])(?:/" + "home" + r"/|/" + "opt" + r"/)[A-Za-z0-9._-]+(?:/|\b)"
    r"|\b[A-Za-z]:\\Users\\[^\\\s]+",
    re.IGNORECASE,
)
_LOCAL_HOST = re.compile(r"\b[A-Za-z0-9-]+\.local\b", re.IGNORECASE)
_STYLE = re.compile(
    r"^(?!\s*%).*\\usepackage\[([^\]]*)\]\{neurips_2026\}",
    re.MULTILINE,
)


def parse_latex_label_page(path: str | Path, label: str) -> int:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{[^{{}}]*\}}\{{(\d+)\}}"
    )
    match = pattern.search(text)
    if match is None:
        raise SubmissionError(f"LaTeX label {label!r} is missing from {path}")
    return int(match.group(1))


def audit_submission_sources(
    main_path: str | Path,
    supplement_path: str | Path,
    checklist_path: str | Path,
    *,
    forbidden_patterns: tuple[str, ...] = (),
) -> dict[str, object]:
    paths = [Path(main_path), Path(supplement_path), Path(checklist_path)]
    errors: list[str] = []
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {
            "status": "fail",
            "errors": ["Missing submission source(s): " + ", ".join(missing)],
        }

    main, supplement, checklist = (
        path.read_text(encoding="utf-8") for path in paths
    )
    style_options = _STYLE.findall(main)
    if not style_options:
        errors.append("The active NeurIPS style must use the eandd track option")
    else:
        parsed_options = [
            {option.strip().lower() for option in raw_options.split(",")}
            for raw_options in style_options
        ]
        if len(parsed_options) != 1:
            errors.append("Submission source has multiple active NeurIPS style declarations")
        if any("eandd" not in options for options in parsed_options):
            errors.append("The active NeurIPS style does not select eandd")
        if any("final" in options for options in parsed_options):
            errors.append("Anonymous submission must not enable the final option")

    if r"\author{Anonymous Authors}" not in main:
        errors.append("Anonymous author declaration is missing")
    for required_input in (r"\input{supplement}", r"\input{checklist}"):
        if required_input not in main:
            errors.append(f"Submission source is missing {required_input}")

    checklist_markers = (
        r"\answerTODO",
        r"\justificationTODO",
        "BEGIN INSTRUCTIONS",
        "END INSTRUCTIONS",
    )
    present_markers = [marker for marker in checklist_markers if marker in checklist]
    if present_markers:
        errors.append(
            "Checklist contains TODO/instruction marker(s): "
            + ", ".join(present_markers)
        )

    combined = f"{main}\n{supplement}\n{checklist}"
    for label, pattern in (
        ("email address", _EMAIL),
        ("local filesystem path", _LOCAL_PATH),
        ("local hostname", _LOCAL_HOST),
    ):
        if pattern.search(combined):
            errors.append(f"Submission source contains a potentially identifying {label}")
    for pattern in forbidden_patterns:
        if pattern and pattern in combined:
            errors.append("Submission source contains a caller-supplied forbidden pattern")

    return {"status": "pass" if not errors else "fail", "errors": errors}


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise SubmissionError(
            f"Command {' '.join(command)!r} failed; see {log_path}"
        )


def build_submission(
    project_root: str | Path,
    *,
    require_final_figures: bool = False,
    forbidden_patterns: tuple[str, ...] = (),
) -> dict[str, object]:
    root = Path(project_root).resolve()
    paper_dir = root / "paper" / "neurips2026"
    main = paper_dir / "main.tex"
    supplement = paper_dir / "supplement.tex"
    checklist = paper_dir / "checklist.tex"
    source_report = audit_submission_sources(
        main,
        supplement,
        checklist,
        forbidden_patterns=forbidden_patterns,
    )
    if source_report["status"] != "pass":
        raise SubmissionError("; ".join(source_report["errors"]))

    required_files = [
        paper_dir / "neurips_2026.sty",
        paper_dir / "evidence_appendix.tex",
        root / "paper" / "references.bib",
        root / "paper" / "citation_audit.tsv",
        root / "paper" / "generated" / "claim_ledger.tsv",
        root / "paper" / "generated" / "claim_values.tex",
        root / "configs" / "audit_rule_catalog.json",
        root / "paper" / "generated" / "submission_closure_v11"
        / "evaluation_method_validation.json",
        root / "paper" / "generated" / "submission_closure_v11"
        / "external_claim_repair.json",
        root / "paper" / "generated" / "submission_closure_v11" / "manifest.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SubmissionError("Missing build input(s): " + ", ".join(missing))

    figure_paths = [
        paper_dir / "figures" / f"figure{index}.pdf" for index in range(1, 6)
    ]
    missing_figures = [path.name for path in figure_paths if not path.is_file()]
    if require_final_figures and missing_figures:
        raise SubmissionError("Missing final figure(s): " + ", ".join(missing_figures))

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None or bibtex is None:
        raise SubmissionError("pdflatex and bibtex must both be available on PATH")

    commands = (
        ([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], "pass1"),
        ([bibtex, "main"], "bibtex"),
        ([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], "pass2"),
        ([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], "pass3"),
    )
    for command, name in commands:
        _run(command, paper_dir, paper_dir / f"build-{name}.log")

    latex_log = (paper_dir / "main.log").read_text(
        encoding="utf-8", errors="replace"
    )
    forbidden_log_messages = (
        "There were undefined citations",
        "There were undefined references",
        r"Overfull \hbox",
    )
    found_log_messages = [
        message for message in forbidden_log_messages if message in latex_log
    ]
    if found_log_messages:
        raise SubmissionError(
            "LaTeX quality gate failed: " + ", ".join(found_log_messages)
        )

    content_pages = parse_latex_label_page(
        paper_dir / "main.aux", "main-content-end"
    )
    if content_pages > 9:
        raise SubmissionError(
            f"Main content occupies {content_pages} pages; E&D permits at most 9"
        )

    bibliography = root / "paper" / "references.bib"
    ledger = root / "paper" / "generated" / "claim_ledger.tsv"
    citation_audit = root / "paper" / "citation_audit.tsv"
    text_audits = {
        "main": audit_manuscript(
            main,
            bibliography,
            ledger,
            citation_audit,
            forbidden_patterns=forbidden_patterns,
            source_root=root,
        ),
        "supplement": audit_manuscript(
            supplement,
            bibliography,
            ledger,
            citation_audit,
            forbidden_patterns=forbidden_patterns,
            source_root=root,
        ),
    }
    if r"\input{evidence_appendix}" in main.read_text(encoding="utf-8"):
        text_audits["evidence_appendix"] = audit_manuscript(
            paper_dir / "evidence_appendix.tex", bibliography, ledger, citation_audit,
            forbidden_patterns=forbidden_patterns, source_root=root,
        )
    if r"\input{reviewer_revision_appendix}" in main.read_text(encoding="utf-8"):
        text_audits["reviewer_revision_appendix"] = audit_manuscript(
            paper_dir / "reviewer_revision_appendix.tex", bibliography, ledger,
            citation_audit, forbidden_patterns=forbidden_patterns, source_root=root,
        )
    failed_text_audits = [
        name for name, audit in text_audits.items() if audit["status"] != "pass"
    ]
    if failed_text_audits:
        raise SubmissionError(
            "Manuscript audit failed for: " + ", ".join(failed_text_audits)
        )

    pdf = paper_dir / "main.pdf"
    if not pdf.is_file():
        raise SubmissionError("LaTeX completed without producing main.pdf")
    report: dict[str, object] = {
        "status": "pass",
        "mode": "final" if require_final_figures else "draft",
        "content_pages": content_pages,
        "page_limit": 9,
        "final_figure_count": 5 - len(missing_figures),
        "missing_final_figures": missing_figures,
        "pdf": {
            "path": str(pdf.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(pdf),
            "size_bytes": pdf.stat().st_size,
        },
        "text_audits": {
            name: {
                "status": audit["status"],
                "citation_count": audit["citation_count"],
                "claim_reference_count": audit["claim_reference_count"],
                "error_count": len(audit["errors"]),
            }
            for name, audit in text_audits.items()
        },
    }
    output = root / "paper" / "generated" / "submission_build_report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
