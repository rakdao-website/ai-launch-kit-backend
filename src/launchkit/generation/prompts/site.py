"""Prompt builders for mockup, planning, page, and v0 generation stages."""

# ruff: noqa: E501

from collections.abc import Mapping, Sequence

from launchkit.generation.prompts.constants import DESIGN_SYSTEM
from launchkit.grounding import FACT_DISCIPLINE
from launchkit.planning.models import PlannedPage, SitePlan


def build_mockup_prompt(direction: str, brief: str, preview_image_url: str | None) -> str:
    image_rule = (
        f"- If you use a hero/background image, use EXACTLY this src: {preview_image_url}\n"
        "  Always ALSO place the headline/subline/buttons as real text on top, so the screen is never blank."
        if preview_image_url
        else "- No photo is available yet for this preview — use a styled gradient/color panel behind the hero text instead of an <img> or external URL."
    )
    return f"""Design ONE hero/landing screen (above-the-fold only) for this brand.
This is a DESIGN MOCKUP for the client to choose a look — make it visually striking and
clearly different from other directions.

{brief}

{FACT_DISCIPLINE}

DESIGN DIRECTION TO FOLLOW:
{direction}

Honor the brand's stated color mood and font feel where it fits, but express THIS direction strongly.

Requirements:
- ONE screen: a nav bar (logo + links) + a hero section. The hero MUST contain VISIBLE TEXT directly
  in the HTML: a real headline, a one-line subline, and at least one CTA button. Write real copy for
  THIS company grounded in the fact sheet — never leave the hero empty or relying only on a background image.
{image_rule}
- The hero section height should be about 600-700px (use min-height:600px) — NOT 100vh.
- Do NOT rely on JavaScript to render content; all text must be present in the static HTML.
- Single self-contained HTML (CSS in <style>). Tailwind via CDN. Google Fonts.
- Make it look like a real, polished website hero — not a wireframe.
- Return ONLY the HTML starting with <!DOCTYPE html>. No markdown, no commentary."""


def build_plan_prompt(
    brief: str,
    chosen_mockup_html: str,
    feedback: str | None = None,
    previous_plan: SitePlan | None = None,
) -> str:
    base = f"""You are planning a marketing website for this specific company.
FIRST choose between 4 and 6 pages that make the most sense for THIS business (page names
tailored to the company — e.g. a bakery might use Menu/Order, a law firm might use Practice
Areas/Case Studies). One page MUST be the homepage. One page MUST be a contact/booking page
(named to fit the brand, e.g. "Book a Demo", "Order", "Contact") whose sections include a
contact/booking form AND the company's location & hours IF those were given in the fact sheet
(if not given, a generic contact form only — do not invent an address). THEN write a concise,
page-by-page plan.

{brief}

{FACT_DISCIPLINE}

CHOSEN DESIGN — the full site must match the look, colors, fonts, nav, and feel of this exact
hero mockup the client picked. Reuse its palette, typography, nav styling, button styling, and
overall aesthetic on every page:
--- CHOSEN MOCKUP HTML START ---
{chosen_mockup_html}
--- CHOSEN MOCKUP HTML END ---

Return ONLY valid JSON in this exact shape:
{{
  "pages": [
    {{"name": "Home", "slug": "index", "isHome": true, "purpose": "one short line on what this page is for",
      "sections": ["2-4 word section", "2-4 word section", "2-4 word section"],
      "images": [{{"section": "which section this photo is for", "desc": "short photorealistic photo description, grounded in the fact sheet — never invent a claim the photo would imply, e.g. don't describe an award plaque that wasn't given"}}]}}
  ]
}}
Each page: a lowercase-hyphenated "slug" (the homepage's slug MUST be "index"), a ONE-LINE
purpose, MAX 3 short section labels (2-4 words each), and 1-2 "images" entries each tied to
ONE named section, with a DIFFERENT specific photo idea per page (never repeat a photo idea
across pages). Keep it skimmable for a non-technical customer. No markdown, no commentary — JSON only."""
    if feedback and previous_plan:
        return f"""{base}

The user reviewed this previous plan:
{previous_plan.raw}

...and asked for these changes:
{feedback}

Revise and return the SAME JSON shape again, keeping one contact/booking page and "images" entries for every page."""
    return base


def build_page_summary(pages: Sequence[PlannedPage]) -> str:
    return "\n".join(
        f"- {page.name} ({page.slug}.html): {'; '.join(page.sections[:4])}" for page in pages
    )


def build_forbidden_sections(pages: Sequence[PlannedPage], current_page_name: str) -> str:
    current = next((page for page in pages if page.name == current_page_name), None)
    own_sections = {section.lower() for section in current.sections} if current else set()
    lines = [
        f"- {section} (belongs to the {page.name} page)"
        for page in pages
        if page.name != current_page_name
        for section in page.sections
        if section.lower() not in own_sections
    ]
    return "\n".join(lines) if lines else "- (none)"


def _theme_implementation(theme: str) -> str:
    chosen = theme or "Light mode"
    compact = "".join(chosen.lower().split())
    if "light+dark" in compact:
        return """THEME IMPLEMENTATION:
- Theme mode is "Light + dark": implement a WORKING toggle — define both themes as CSS variables on
  :root and [data-theme="dark"], add a visible toggle button in the nav whose click flips
  document.documentElement.dataset.theme, persist the choice with localStorage, and default to the
  user's prefers-color-scheme. Every color on the page must come from the variables so BOTH themes
  fully work. A toggle that does nothing is a failure."""
    mode = "dark" if "dark" in chosen.lower() else "light"
    return f"""THEME IMPLEMENTATION:
- Theme mode is single-mode: build ONLY a {mode} theme and do NOT show any theme toggle."""


def build_page_prompt(
    *,
    page: PlannedPage,
    is_home: bool,
    brief: str,
    chosen_mockup_html: str,
    site_map: str,
    image_catalog: str,
    cta_rule: str,
    nav_html: str,
    footer_html: str,
    forbidden_sections: str,
    is_order_page: bool,
    theme: str,
) -> str:
    order_requirement = (
        """

THIS IS THE CONTACT/BOOKING PAGE: it MUST contain a working contact/booking form (name, email,
message or preferred date, with JS validation and a visible success state). Include a location &
hours section ONLY with details from the fact sheet; if none were provided, show the form plus a
generic "reach out" line — never invent an address, phone number, or opening hours."""
        if is_order_page
        else ""
    )
    shared = f"""
{brief}

{FACT_DISCIPLINE}

{DESIGN_SYSTEM}

{_theme_implementation(theme)}

IMAGES:
{image_catalog}

WHOLE-SITE MAP (every page and what it covers — do NOT put another page's content on this page):
{site_map}

STRICTLY FORBIDDEN ON THIS PAGE — these topics belong to OTHER pages and must NOT appear here
(no pricing tables, membership tiers, forms, galleries, etc. unless listed in THIS page's own sections):
{forbidden_sections}{order_requirement}

CALL-TO-ACTION RULE:
{cta_rule}

SEO REQUIREMENTS:
- A unique, descriptive <title> for this page.
- A <meta name="description"> (max 155 chars) summarizing THIS page for search results, using only facts from the fact sheet.
- Open Graph tags: og:title, og:description, og:type ("website"), and og:image if this page has an image.
- One single <h1> per page; use h2/h3 for the rest. Descriptive alt text on every image.

Reuse the exact Google Fonts and color palette from the chosen mockup so all pages match.
Output a SINGLE self-contained HTML file (CSS in <style>, JS in <script>).
Return ONLY the HTML starting with <!DOCTYPE html>. No markdown, no commentary."""
    sections = "\n".join(f"- {section}" for section in page.sections)
    if is_home:
        return f"""Build the HOMEPAGE ("{page.name}", file: index.html) of a multi-page site.
{shared}

This page's sections (build ONLY these — this is the homepage, so it gets the main hero):
{sections}

CHOSEN DESIGN — match this exact hero mockup's look, colors, fonts, nav, and feel:
--- CHOSEN MOCKUP HTML START ---
{chosen_mockup_html}
--- CHOSEN MOCKUP HTML END ---

NAV: a sticky nav linking to every page in the site map above by exact file name. Highlight "{page.name}" as active. Include a matching footer."""

    return f"""Build the "{page.name}" page (file: {page.slug}.html) of a multi-page site.
{shared}

CRITICAL RULES:
- Build ONLY this page's own sections (listed below). Do NOT add sections that belong to other
  pages. This is its OWN dedicated page.
- This is an INNER page, NOT the homepage: do NOT copy the homepage's big hero. Use a smaller,
  distinct page-header for this page, then this page's unique content with its OWN layout.
- Make the layout visibly different from the other pages while keeping the same colors/fonts.

This page's sections (build exactly these):
{sections}

USE THIS EXACT NAV (paste it as-is, but change which link is marked active to "{page.name}"):
{nav_html}

USE THIS EXACT FOOTER (paste it as-is):
{footer_html}"""


def build_v0_multi_page_brief(
    brief: str,
    chosen_mockup_html: str,
    pages: Sequence[PlannedPage],
    image_catalog_by_page: Mapping[str, str],
) -> str:
    page_specs = "\n\n".join(
        f"""### {page.name} ({"home page" if page.is_home else page.slug})
Purpose: {page.purpose}
Sections: {", ".join(page.sections)}
{image_catalog_by_page.get(page.name, "No images assigned — use styled color panels instead.")}"""
        for page in pages
    )
    return f"""Build a polished, multi-page marketing website as a Next.js app.

{brief}

{FACT_DISCIPLINE}

CHOSEN DESIGN DIRECTION — match this hero mockup's look, palette, typography, and overall feel
across every page (don't copy it verbatim, follow its aesthetic):
--- CHOSEN MOCKUP HTML START ---
{chosen_mockup_html}
--- CHOSEN MOCKUP HTML END ---

{DESIGN_SYSTEM}

PAGES TO BUILD (build exactly these, each as its own route):
{page_specs}

Every page shares one sticky nav (linking to all pages above) and one footer. Every primary
call-to-action links to the contact/booking/order page. Do not leave any button non-functional.
Every nav link must point only to a page you built — never leave a route that returns 404; if
content is thin, use a simple placeholder page instead of a broken route.
Do not invent stats, testimonials, addresses, or team members beyond the fact sheet above — if a
conventional section (like testimonials) has no real content in the fact sheet, omit it rather
than inventing one."""
