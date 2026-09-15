"""Merge stored business drafts with extracted profile fields for generation."""

from __future__ import annotations

from typing import Any, Mapping

from launchkit.projects.models import BusinessDraft


def merge_empty(current: dict[str, object], extracted: dict[str, object]) -> dict[str, object]:
    """Fill empty current keys from extracted values without overwriting edits."""

    merged = dict(current)
    for key, value in extracted.items():
        existing = merged.get(key)
        if value is not None and value != "" and (existing is None or existing == ""):
            merged[key] = value
    return merged


def business_form_for_generation(
    business: Mapping[str, Any] | None,
    extracted_profile_fields: Mapping[str, Any] | None,
    *,
    category_fallback_industry: str | None = None,
) -> BusinessDraft:
    """Build the grounded onboarding form used by mockups and final builds.

    User-edited ``business`` values win. Empty business slots are filled from
    ``extracted_profile_fields`` so PDF/site discovery facts (products, contact,
    hours, testimonials, etc.) still reach the fact sheet when the AI Summary
    modal only surface-edited a few fields.
    """

    merged = merge_empty(dict(business or {}), dict(extracted_profile_fields or {}))
    form = BusinessDraft.model_validate(merged)
    if not form.industry and category_fallback_industry:
        form = form.model_copy(update={"industry": category_fallback_industry})
    return form
