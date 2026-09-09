"""Deterministic citation, claim-reference, and anonymity checks for manuscripts."""

from __future__ import annotations

import csv
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

_BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_CITATION = re.compile(r"(?<![\w@])@([A-Za-z][A-Za-z0-9_:-]*)")
_LATEX_CITATION = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar)"
    r"\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]+)\}",
    re.IGNORECASE,
)
_CLAIM_BLOCK = re.compile(r"<!--\s*claims:\s*([^>]+?)\s*-->", re.IGNORECASE)
_LATEX_CLAIM_BLOCK = re.compile(
    r"%+\s*claims:\s*([^\r\n]+)", re.IGNORECASE
)
_LATEX_CLAIM_VALUE = re.compile(
    r"\\claimvalue\s*\{\s*(C\d+)\s*\}", re.IGNORECASE
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_UNIX_HOME = re.compile(r"(?<![\w.])/" + "home" + r"/[A-Za-z0-9._-]+(?:/|\b)")
_UNIX_OPT = re.compile(r"(?<![\w.])/" + "opt" + r"/[A-Za-z0-9._-]+(?:/|\b)")
_WINDOWS_USER = re.compile(
    r"\b[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+", re.IGNORECASE
)
_LOCAL_HOST = re.compile(r"\b[A-Za-z0-9-]+\.local\b", re.IGNORECASE)
_NUMERIC_SOURCE = re.compile(
    r"<!--\s*(?:claims|constants-source|evidence):\s*[^>]+-->",
    re.IGNORECASE,
)
_SOURCE_BLOCK = re.compile(
    r"<!--\s*(?:constants-source|evidence):\s*([^>]+?)\s*-->",
    re.IGNORECASE,
)
_LATEX_SOURCE_BLOCK = re.compile(
    r"%+\s*(?:constants-source|evidence):\s*([^\r\n]+)", re.IGNORECASE
)
_REPOSITORY_SOURCE_PATH = re.compile(
    r"\b(?:cards|configs|docs|paper|scripts|src)/[A-Za-z0-9_./*-]+"
)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_CITATION_BLOCK = re.compile(r"\[[^\]]*@[^\]]+\]")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-Za-z0-9])"
)
_CLAIM_VALUE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})"
    r"(?![A-Za-z0-9%])"
)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def _read_tsv_column(path: Path, column: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} is missing required column {column!r}")
        return [str(row[column]).strip() for row in reader if str(row[column]).strip()]


def _claim_display_values(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"claim_id", "display_value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain claim_id and display_value columns")
        return {
            str(row["claim_id"]).strip(): str(row["display_value"]).strip()
            for row in reader
            if str(row["claim_id"]).strip()
        }


def _claim_allowed_numeric_values(path: Path) -> dict[str, set[Decimal]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"claim_id", "display_value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain claim_id and display_value columns")
        values: dict[str, set[Decimal]] = {}
        for row in reader:
            claim_id = str(row["claim_id"]).strip()
            if not claim_id:
                continue
            candidates = [str(row["display_value"]).strip()]
            if "allowed_wording" in row:
                candidates.extend(
                    _CLAIM_VALUE_NUMBER.findall(str(row["allowed_wording"]))
                )
            values[claim_id] = {
                normalized
                for candidate in candidates
                for normalized in [_normalized_decimal(candidate)]
                if normalized is not None
            }
        return values


def extract_bibliography_keys(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return _BIB_ENTRY.findall(text)


def extract_citation_keys(text: str) -> list[str]:
    references = _CITATION.findall(text)
    for block in _LATEX_CITATION.findall(text):
        references.extend(
            token.strip()
            for token in block.split(",")
            if token.strip()
        )
    return references


def extract_claim_references(text: str) -> list[str]:
    references = extract_explicit_claim_references(text)
    references.extend(_LATEX_CLAIM_VALUE.findall(text))
    return references


def extract_explicit_claim_references(text: str) -> list[str]:
    references: list[str] = []
    for pattern in (_CLAIM_BLOCK, _LATEX_CLAIM_BLOCK):
        for match in pattern.finditer(text):
            references.extend(
                token
                for token in re.split(r"[\s,;]+", match.group(1).strip())
                if token
            )
    return references


def _block_has_citation(text: str) -> bool:
    return bool(_CITATION.search(text) or _LATEX_CITATION.search(text))


def _block_has_numeric_source(text: str) -> bool:
    return bool(
        _NUMERIC_SOURCE.search(text)
        or _LATEX_SOURCE_BLOCK.search(text)
        or _LATEX_CLAIM_BLOCK.search(text)
        or _LATEX_CLAIM_VALUE.search(text)
    )


def extract_source_references(text: str) -> list[str]:
    references: list[str] = []
    for pattern in (_SOURCE_BLOCK, _LATEX_SOURCE_BLOCK):
        for block in pattern.findall(text):
            references.extend(_REPOSITORY_SOURCE_PATH.findall(block))
    return references


def _clean_numeric_block(block: str) -> str:
    cleaned_lines: list[str] = []
    for line in block.splitlines():
        if line.lstrip().startswith("#"):
            continue
        cleaned_lines.append(re.sub(r"^\s*\d+[.)]\s+", "", line))
    cleaned = "\n".join(cleaned_lines)
    cleaned = _INLINE_CODE.sub("", cleaned)
    cleaned = _CITATION_BLOCK.sub("", cleaned)
    cleaned = _HTML_COMMENT.sub("", cleaned)
    cleaned = re.sub(r"\bSHA-?\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:Figure|Fig\.?|Table)\s+\d+\b|(?:图|表)\s*\d+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def audit_numeric_evidence_blocks(text: str) -> tuple[int, list[dict[str, object]]]:
    line_offset = 0
    document_marker = r"\begin{document}"
    if document_marker in text:
        prefix, text = text.split(document_marker, maxsplit=1)
        line_offset = prefix.count("\n")
    numeric_block_count = 0
    unmapped: list[dict[str, object]] = []
    block_matches = re.finditer(
        r"(?:\A|\n\s*\n)(.*?)(?=\n\s*\n|\Z)",
        text,
        re.DOTALL,
    )
    for block_index, match in enumerate(block_matches, start=1):
        block = match.group(1)
        if not _NUMBER.search(_clean_numeric_block(block)):
            continue
        numeric_block_count += 1
        if _block_has_numeric_source(block) or _block_has_citation(block):
            continue
        line_number = line_offset + text.count("\n", 0, match.start(1)) + 1
        excerpt = " ".join(block.split())
        unmapped.append(
            {
                "block": block_index,
                "line": line_number,
                "excerpt": excerpt[:180],
            }
        )
    return numeric_block_count, unmapped


def _normalized_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def audit_claim_display_values(
    text: str,
    display_values: dict[str, str],
    allowed_numeric_values: dict[str, set[Decimal]],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    block_matches = re.finditer(
        r"(?:\A|\n\s*\n)(.*?)(?=\n\s*\n|\Z)",
        text,
        re.DOTALL,
    )
    for match in block_matches:
        block = match.group(1)
        claim_ids = extract_explicit_claim_references(block)
        if not claim_ids:
            continue
        allowed = {
            normalized
            for claim_id in claim_ids
            for normalized in allowed_numeric_values.get(claim_id, set())
        }
        for raw_value in _CLAIM_VALUE_NUMBER.findall(_clean_numeric_block(block)):
            normalized = _normalized_decimal(raw_value)
            if normalized is None or normalized in allowed:
                continue
            mismatches.append(
                {
                    "line": text.count("\n", 0, match.start(1)) + 1,
                    "value": raw_value,
                    "claim_ids": sorted(set(claim_ids)),
                    "allowed_display_values": sorted(
                        {
                            display_values[claim_id]
                            for claim_id in claim_ids
                            if claim_id in display_values
                        }
                    ),
                }
            )
    return mismatches


def _citation_audit(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cite_key", "status"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain cite_key and status columns")
        rows = [row for row in reader]
    keys = [str(row["cite_key"]).strip() for row in rows]
    statuses = {
        str(row["cite_key"]).strip(): str(row["status"]).strip()
        for row in rows
        if str(row["cite_key"]).strip()
    }
    return keys, statuses


def audit_manuscript(
    manuscript_path: str | Path,
    bibliography_path: str | Path,
    claim_ledger_path: str | Path,
    citation_audit_path: str | Path,
    *,
    forbidden_patterns: tuple[str, ...] = (),
    source_root: str | Path | None = None,
) -> dict[str, object]:
    manuscript = Path(manuscript_path)
    bibliography = Path(bibliography_path)
    claim_ledger = Path(claim_ledger_path)
    citation_audit = Path(citation_audit_path)

    text = manuscript.read_text(encoding="utf-8")
    citation_references = extract_citation_keys(text)
    claim_references = extract_claim_references(text)
    bibliography_keys = extract_bibliography_keys(bibliography)
    ledger_keys = _read_tsv_column(claim_ledger, "claim_id")
    claim_display_values = _claim_display_values(claim_ledger)
    claim_allowed_numeric_values = _claim_allowed_numeric_values(claim_ledger)
    audited_keys, audit_statuses = _citation_audit(citation_audit)
    numeric_block_count, unmapped_numeric_blocks = audit_numeric_evidence_blocks(text)
    claim_value_mismatches = audit_claim_display_values(
        text,
        claim_display_values,
        claim_allowed_numeric_values,
    )
    source_references = extract_source_references(text)
    if source_root is None:
        source_root_path = (
            manuscript.parent.parent
            if manuscript.parent.name == "paper"
            else manuscript.parent
        )
    else:
        source_root_path = Path(source_root)
    missing_source_references = sorted(
        {
            reference
            for reference in source_references
            if not (
                any(source_root_path.glob(reference))
                if "*" in reference
                else (source_root_path / reference).exists()
            )
        }
    )

    errors: list[str] = []
    for label, values in (
        ("bibliography keys", bibliography_keys),
        ("claim ledger IDs", ledger_keys),
        ("citation audit keys", audited_keys),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            errors.append(f"Duplicate {label}: {', '.join(duplicates)}")

    missing_bibliography = sorted(set(citation_references) - set(bibliography_keys))
    if missing_bibliography:
        errors.append(
            "Citations missing from bibliography: " + ", ".join(missing_bibliography)
        )

    missing_citation_audit = sorted(set(citation_references) - set(audited_keys))
    if missing_citation_audit:
        errors.append(
            "Citations missing from citation audit: "
            + ", ".join(missing_citation_audit)
        )

    bibliography_without_audit = sorted(set(bibliography_keys) - set(audited_keys))
    if bibliography_without_audit:
        errors.append(
            "Bibliography entries missing from citation audit inventory: "
            + ", ".join(bibliography_without_audit)
        )

    audit_without_bibliography = sorted(set(audited_keys) - set(bibliography_keys))
    if audit_without_bibliography:
        errors.append(
            "Citation audit entries missing from bibliography: "
            + ", ".join(audit_without_bibliography)
        )

    unresolved_status = sorted(
        key
        for key in set(citation_references) & set(audited_keys)
        if not audit_statuses.get(key, "").startswith("verified")
    )
    if unresolved_status:
        errors.append(
            "Citations lack a verified citation audit status: "
            + ", ".join(unresolved_status)
        )

    missing_claims = sorted(set(claim_references) - set(ledger_keys))
    if missing_claims:
        errors.append("Claim references missing from ledger: " + ", ".join(missing_claims))

    if unmapped_numeric_blocks:
        lines = ", ".join(str(item["line"]) for item in unmapped_numeric_blocks[:12])
        suffix = " ..." if len(unmapped_numeric_blocks) > 12 else ""
        errors.append(
            "Manuscript contains numeric block(s) without a claim, constant source, "
            f"generated-evidence marker, or audited citation at lines: {lines}{suffix}"
        )

    if missing_source_references:
        errors.append(
            "Manuscript source reference(s) do not resolve under the source root: "
            + ", ".join(missing_source_references)
        )

    if claim_value_mismatches:
        locations = ", ".join(
            f"line {item['line']}: {item['value']}"
            for item in claim_value_mismatches[:12]
        )
        suffix = " ..." if len(claim_value_mismatches) > 12 else ""
        errors.append(
            "Numeric value(s) do not match any referenced claim display value: "
            f"{locations}{suffix}"
        )

    sensitive_checks = (
        ("email address", _EMAIL),
        ("local home path", _UNIX_HOME),
        ("local opt path", _UNIX_OPT),
        ("Windows user path", _WINDOWS_USER),
        ("local hostname", _LOCAL_HOST),
    )
    for label, pattern in sensitive_checks:
        if pattern.search(text):
            errors.append(f"Manuscript contains a potentially identifying {label}")
    for pattern in forbidden_patterns:
        if pattern and pattern in text:
            errors.append("Manuscript contains a caller-supplied forbidden pattern")

    return {
        "status": "pass" if not errors else "fail",
        "manuscript": manuscript.name,
        "citation_count": len(set(citation_references)),
        "claim_reference_count": len(set(claim_references)),
        "bibliography_entry_count": len(bibliography_keys),
        "citation_audit_entry_count": len(audited_keys),
        "ledger_claim_count": len(ledger_keys),
        "numeric_evidence_block_count": numeric_block_count,
        "unmapped_numeric_block_count": len(unmapped_numeric_blocks),
        "unmapped_numeric_blocks": unmapped_numeric_blocks,
        "source_reference_count": len(set(source_references)),
        "source_references": sorted(set(source_references)),
        "missing_source_references": missing_source_references,
        "claim_value_mismatch_count": len(claim_value_mismatches),
        "claim_value_mismatches": claim_value_mismatches,
        "citations": sorted(set(citation_references)),
        "claim_references": sorted(set(claim_references)),
        "unused_bibliography_entries": sorted(
            set(bibliography_keys) - set(citation_references)
        ),
        "unreferenced_ledger_claims": sorted(set(ledger_keys) - set(claim_references)),
        "errors": errors,
    }
