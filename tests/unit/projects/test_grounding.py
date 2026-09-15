"""Tests for merging extracted profile fields into generation forms."""

from launchkit.projects.grounding import business_form_for_generation, merge_empty


def test_merge_empty_fills_only_blank_slots() -> None:
    merged = merge_empty(
        {"companyName": "Velora", "contact": ""},
        {"companyName": "Other", "contact": "+971 4 555 2847", "products": "Caspian Weekender"},
    )
    assert merged["companyName"] == "Velora"
    assert merged["contact"] == "+971 4 555 2847"
    assert merged["products"] == "Caspian Weekender"


def test_business_form_for_generation_merges_extract_into_thin_business() -> None:
    form = business_form_for_generation(
        {
            "companyName": "Velora Atelier",
            "categoryId": "retail-ecommerce",
            "uvp": "Hand-finished leather for the Gulf",
            "targetAudience": "Professionals in Dubai",
        },
        {
            "products": "Caspian Weekender AED 2,480; Oasis Card Sleeve AED 320",
            "contact": "+971 4 555 2847 · hello@velora-atelier.ae",
            "locationHours": "Unit 12, Marina Walk Arcade; Fri 14:00–20:00",
            "testimonials": "Maya Al-Hassan: repaired in three days",
            "teamBios": "Lina Haddad — Creative Director",
            "stats": "4,200+ numbered pieces",
            "tone": "Warm, precise, unhurried",
        },
        category_fallback_industry="Retail",
    )

    assert form.company_name == "Velora Atelier"
    assert form.uvp.startswith("Hand-finished")
    assert "Caspian Weekender" in form.products
    assert "+971 4 555 2847" in form.contact
    assert "Marina Walk" in form.location_hours
    assert "Maya Al-Hassan" in form.testimonials
    assert "Lina Haddad" in form.team_bios
    assert "4,200" in form.stats
    assert form.tone.startswith("Warm")
    assert form.industry == "Retail"


def test_business_form_ignores_design_hint_keys_from_extract() -> None:
    form = business_form_for_generation(
        {"companyName": "Velora Atelier", "categoryId": "retail-ecommerce"},
        {
            "products": "Caspian Weekender",
            "tagline": "Hand-finished leather, made for the Gulf pace of life",
            "cta": "Book a fitting",
        },
    )
    assert form.company_name == "Velora Atelier"
    assert "Caspian Weekender" in form.products
