import hashlib
import json
from pathlib import Path

from launchkit.design import DesignPreferences, resolve_industry_style_direction
from launchkit.generation.prompts import (
    DESIGN_SYSTEM,
    build_brief,
    build_forbidden_sections,
    build_legacy_brief_user_message,
    build_mockup_prompt,
    build_page_prompt,
    build_page_summary,
    build_plan_prompt,
    build_v0_multi_page_brief,
)
from launchkit.grounding import FACT_DISCIPLINE
from launchkit.intake import LegacyCompany, OnboardingForm
from launchkit.planning import PlannedPage, SitePlan

FIXTURES = Path(__file__).parents[2] / "fixtures"


def _load_brief_inputs() -> tuple[OnboardingForm, DesignPreferences]:
    form_payload = json.loads((FIXTURES / "haseeb_intake.json").read_text(encoding="utf-8"))
    design_payload = json.loads((FIXTURES / "haseeb_design.json").read_text(encoding="utf-8"))
    return (
        OnboardingForm.model_validate(form_payload),
        DesignPreferences.model_validate(design_payload),
    )


def _pages() -> list[PlannedPage]:
    return [
        PlannedPage(
            name="Home",
            slug="index",
            is_home=True,
            purpose="Introduce Northstar",
            sections=["Hero", "Services", "Proof"],
            images=[],
        ),
        PlannedPage(
            name="Book a Demo",
            slug="book-a-demo",
            is_home=False,
            purpose="Capture qualified leads",
            sections=["Contact Form", "Location Hours"],
            images=[],
        ),
    ]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_brief_combines_grounding_design_tokens_and_industry_direction() -> None:
    form, design = _load_brief_inputs()
    direction = resolve_industry_style_direction(form)

    brief = build_brief(form, design, direction)

    assert brief.startswith("FACT SHEET (the only source of truth")
    assert "Company: Northstar Labs" in brief
    assert "- Visual style: Modern / minimal" in brief
    assert "- Primary color: #3B6FED" in brief
    assert '- Heading font: "Poppins"' in brief
    assert brief.endswith(direction)


def test_brief_preserves_empty_design_fallbacks() -> None:
    brief = build_brief(OnboardingForm(), DesignPreferences(), "")

    assert "IDENTITY: NOT PROVIDED" in brief
    assert "Let the AI choose to fit the brand" in brief
    assert "(none given — write one grounded in the UVP above)" in brief
    assert "EXACT DESIGN TOKENS" not in brief


def test_legacy_brief_message_matches_root_template_shape() -> None:
    company = LegacyCompany(
        name="Acme",
        industry="Consulting",
        tagline="Clarity first",
        description="Advises operators",
        services="Strategy, delivery",
        audience="Founders",
        tone="Direct",
        location="Dubai",
        website="https://acme.example",
        contact_email="hello@acme.example",
        contact_phone="+971 555 0100",
        colorway="navy and gold",
        animation_level="moderate",
    )

    message = build_legacy_brief_user_message(company)

    assert message.startswith("Convert the business information below")
    assert "<user_business_data>\nCompany name: Acme" in message
    assert "Preferred animation level: moderate\n\n</user_business_data>" in message


def test_mockup_prompt_handles_assigned_and_missing_preview_images() -> None:
    with_image = build_mockup_prompt("Editorial", "BRIEF", "/hero.jpg")
    without_image = build_mockup_prompt("Editorial", "BRIEF", None)

    assert FACT_DISCIPLINE in with_image
    assert "use EXACTLY this src: /hero.jpg" in with_image
    assert "Always ALSO place" in with_image
    assert "No photo is available yet" in without_image
    assert "min-height:600px" in without_image


def test_plan_prompt_supports_initial_and_revision_flows() -> None:
    pages = _pages()
    plan = SitePlan(pages=pages, raw='{"pages": []}')

    initial = build_plan_prompt("BRIEF", "<main>chosen</main>")
    revised = build_plan_prompt("BRIEF", "<main>chosen</main>", "Add FAQ", plan)
    feedback_without_plan = build_plan_prompt("BRIEF", "mockup", "Add FAQ")

    assert '"slug": "index"' in initial
    assert "The user reviewed this previous plan" not in initial
    assert '{"pages": []}' in revised
    assert "Add FAQ" in revised
    assert feedback_without_plan == build_plan_prompt("BRIEF", "mockup")


def test_page_summary_and_forbidden_sections_match_plan_ownership() -> None:
    pages = _pages()

    assert build_page_summary(pages) == (
        "- Home (index.html): Hero; Services; Proof\n"
        "- Book a Demo (book-a-demo.html): Contact Form; Location Hours"
    )
    assert build_forbidden_sections(pages, "Home") == (
        "- Contact Form (belongs to the Book a Demo page)\n"
        "- Location Hours (belongs to the Book a Demo page)"
    )
    assert build_forbidden_sections(pages, "Missing").startswith(
        "- Hero (belongs to the Home page)"
    )
    assert build_forbidden_sections([pages[0]], "Home") == "- (none)"


def test_forbidden_sections_does_not_repeat_shared_current_sections() -> None:
    pages = _pages()
    pages[1].sections.append("Hero")

    forbidden = build_forbidden_sections(pages, "Home")

    assert "Book a Demo page)\n- Hero" not in forbidden


def test_page_prompts_keep_home_inner_contact_and_theme_contracts() -> None:
    home, contact = _pages()
    common = {
        "brief": "BRIEF",
        "chosen_mockup_html": "<main>chosen</main>",
        "site_map": "Home, Book a Demo",
        "image_catalog": "No images",
        "cta_rule": "All CTAs go to book-a-demo.html",
        "nav_html": "<nav>shared</nav>",
        "footer_html": "<footer>shared</footer>",
        "forbidden_sections": "- (none)",
    }

    home_prompt = build_page_prompt(
        page=home, is_home=True, is_order_page=False, theme="", **common
    )
    contact_prompt = build_page_prompt(
        page=contact,
        is_home=False,
        is_order_page=True,
        theme="Light + dark (theme toggle)",
        **common,
    )
    dark_prompt = build_page_prompt(
        page=contact, is_home=False, is_order_page=False, theme="Dark mode", **common
    )

    assert home_prompt.startswith('Build the HOMEPAGE ("Home", file: index.html)')
    assert "build ONLY a light theme" in home_prompt
    assert "CHOSEN MOCKUP HTML START" in home_prompt
    assert "THIS IS THE CONTACT/BOOKING PAGE" in contact_prompt
    assert "implement a WORKING toggle" in contact_prompt
    assert "USE THIS EXACT NAV" in contact_prompt
    assert "build ONLY a dark theme" in dark_prompt
    assert DESIGN_SYSTEM in dark_prompt


def test_v0_brief_renders_assigned_and_fallback_image_catalogs() -> None:
    pages = _pages()

    prompt = build_v0_multi_page_brief(
        "BRIEF",
        "<main>chosen</main>",
        pages,
        {"Home": "Hero: /hero.jpg"},
    )

    assert "### Home (home page)" in prompt
    assert "Hero: /hero.jpg" in prompt
    assert "### Book a Demo (book-a-demo)" in prompt
    assert "No images assigned — use styled color panels instead." in prompt
    assert "build exactly these, each as its own route" in prompt
    assert "never leave a route that returns 404" in prompt


def test_full_prompt_outputs_match_characterization_digests() -> None:
    form, design = _load_brief_inputs()
    brief = build_brief(form, design, resolve_industry_style_direction(form))
    pages = _pages()
    common = {
        "brief": brief,
        "chosen_mockup_html": "<main>chosen</main>",
        "site_map": "Home, Book a Demo",
        "image_catalog": "Hero: /hero.jpg",
        "cta_rule": "All CTAs go to book-a-demo.html",
        "nav_html": "<nav>shared</nav>",
        "footer_html": "<footer>shared</footer>",
        "forbidden_sections": "- Contact Form (belongs to the Book a Demo page)",
    }
    outputs = {
        "brief": brief,
        "mockup": build_mockup_prompt("Editorial", brief, "/hero.jpg"),
        "plan": build_plan_prompt(brief, "<main>chosen</main>"),
        "home": build_page_prompt(
            page=pages[0], is_home=True, is_order_page=False, theme="Light mode", **common
        ),
        "inner": build_page_prompt(
            page=pages[1],
            is_home=False,
            is_order_page=True,
            theme="Light + dark (theme toggle)",
            **common,
        ),
        "v0": build_v0_multi_page_brief(
            brief, "<main>chosen</main>", pages, {"Home": "Hero: /hero.jpg"}
        ),
    }
    expected = {
        "brief": "b8dbac1d2625220a35afb4ed49bc6c209cc3328789ece361e6b09b7d2c7a9315",
        "mockup": "8c4f7ffe45a9974c1428c7664e17acc4d0d4b305da530c81932a961d34a125a3",
        "plan": "e3691bf101904ed3d50603cc84d63e647b7824c791330f88e954b76ed5618fb0",
        "home": "f9da7a8ea5dbd454bcceae19b3d033b622132c613826e92bf3b9347a114667ca",
        "inner": "921c4024007244c4a1fe5c6863304e844998127adf8a0f7531216ce2454943c9",
        "v0": "7e844f6bd21cde2a92d9d3975b514e4d059d373bdae11fe551d3cffc97e79858",
    }

    assert {name: _digest(value) for name, value in outputs.items()} == expected
