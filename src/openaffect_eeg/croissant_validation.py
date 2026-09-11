"""Offline validation for the OpenAffect-EEG Croissant core-plus-RAI profile."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


class CroissantValidationError(ValueError):
    """Raised when the release metadata is incomplete or unsafe to submit."""


CORE_FIELDS = {
    "@context", "@type", "name", "description", "license", "url",
    "conformsTo", "creator", "dateCreated", "dateModified", "distribution",
    "recordSet",
}
RAI_FIELDS = {
    "rai:dataCollection", "rai:dataCollectionType", "rai:dataCollectionRawData",
    "rai:dataCollectionMissingData", "rai:dataAnnotationAnalysis", "rai:dataUseCases",
    "rai:dataLimitations", "rai:dataBiases", "rai:personalSensitiveInformation",
    "rai:dataSocialImpact", "rai:dataReleaseMaintenancePlan",
}
RAI_CONFORMS_TO = "http://mlcommons.org/croissant/RAI/1.0"


def _https_url(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise CroissantValidationError(f"{field} must be an HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise CroissantValidationError(f"{field} must be an HTTPS URL")
    return value


def _placeholder_url(url: str) -> bool:
    return urlparse(url).hostname == "replace-before-submission.invalid"


def validate_croissant_metadata(
    metadata: dict[str, Any],
    *,
    require_publishable_url: bool = False,
) -> dict[str, object]:
    """Validate core fields and the project's documented Croissant RAI profile."""
    if not isinstance(metadata, dict):
        raise CroissantValidationError("Croissant metadata must be a JSON object")
    missing_core = sorted(CORE_FIELDS.difference(metadata))
    if missing_core:
        raise CroissantValidationError(f"Croissant is missing core fields: {missing_core}")
    missing_rai = sorted(RAI_FIELDS.difference(metadata))
    if missing_rai:
        raise CroissantValidationError(f"Croissant is missing RAI fields: {missing_rai}")
    context = metadata["@context"]
    if not isinstance(context, dict) or context.get("rai") != "http://mlcommons.org/croissant/RAI/":
        raise CroissantValidationError("Croissant context must declare the RAI namespace")
    conforms_to = metadata["conformsTo"]
    declared = {conforms_to} if isinstance(conforms_to, str) else set(conforms_to)
    if RAI_CONFORMS_TO not in declared:
        raise CroissantValidationError("Croissant must declare the RAI conformance URI")
    canonical_url = _https_url(metadata["url"], field="url")
    distributions = metadata["distribution"]
    if not isinstance(distributions, list) or not distributions:
        raise CroissantValidationError("Croissant distribution must be a non-empty list")
    distribution_urls: list[str] = []
    for index, distribution in enumerate(distributions):
        if not isinstance(distribution, dict):
            raise CroissantValidationError("Croissant distributions must be objects")
        distribution_urls.append(
            _https_url(distribution.get("contentUrl"), field=f"distribution[{index}].contentUrl")
        )
    source_urls = [url for url in distribution_urls if "openneuro.org/datasets/" in url]
    release_urls = [
        url for url in distribution_urls
        if url.endswith((".tar.gz", ".zip", ".csv"))
        and "openneuro.org/datasets/" not in url
    ]
    if not source_urls:
        raise CroissantValidationError("Croissant must retain at least one OpenNeuro provenance URL")
    if not release_urls:
        raise CroissantValidationError("Croissant must describe a released anonymous artifact")
    record_sets = metadata["recordSet"]
    if not isinstance(record_sets, list) or not record_sets:
        raise CroissantValidationError("Croissant recordSet must be a non-empty list")
    for record_set in record_sets:
        if not isinstance(record_set, dict) or not record_set.get("field"):
            raise CroissantValidationError("Each Croissant record set must describe fields")
    placeholder = any(_placeholder_url(url) for url in [canonical_url, *release_urls])
    if require_publishable_url and placeholder:
        raise CroissantValidationError(
            "Croissant requires a publishable anonymous code/archive URL before submission"
        )
    return {
        "status": "valid",
        "record_set_count": len(record_sets),
        "distribution_count": len(distributions),
        "source_distribution_count": len(source_urls),
        "release_archive_distribution_count": len(release_urls),
        "rai_field_count": len(RAI_FIELDS),
        "release_url_status": "placeholder" if placeholder else "publishable",
    }
