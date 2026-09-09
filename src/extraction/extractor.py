"""
extractor.py
------------
Extracts structured signals from raw résumé / job-offer text:

- `find_skills`      -> canonical skills detected, grouped by category
- `extract_experience_years`  -> years of professional experience mentioned
- `extract_education_level`   -> highest education level mentioned (0-5 scale)

All matching is case-insensitive and accent-insensitive (so "développeur"
and "developpeur" both match), and uses word boundaries via regex to avoid
partial-word false positives (e.g. "java" should not match inside "javascript"
unless intended - handled with negative lookahead in skills_data.py).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .skills_data import all_skills_flat, skill_category

# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase and strip accents, so matching is robust to accent variants."""
    text = text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# --------------------------------------------------------------------------
# Skill extraction
# --------------------------------------------------------------------------

_SKILL_PATTERNS_CACHE: dict[str, re.Pattern] | None = None


def _compiled_patterns() -> dict[str, re.Pattern]:
    """Compile one combined regex per canonical skill (cached across calls)."""
    global _SKILL_PATTERNS_CACHE
    if _SKILL_PATTERNS_CACHE is None:
        compiled = {}
        for skill, patterns in all_skills_flat().items():
            normalized_patterns = [normalize(p) for p in patterns]
            combined = "|".join(f"(?:{p})" for p in normalized_patterns)
            compiled[skill] = re.compile(combined, flags=re.IGNORECASE)
        _SKILL_PATTERNS_CACHE = compiled
    return _SKILL_PATTERNS_CACHE


def find_skills(text: str) -> dict[str, list[str]]:
    """
    Detect canonical skills present in `text`.

    Returns:
        { category: [canonical_skill, ...] } for every skill found at least once.
    """
    normalized_text = normalize(text)
    found_by_category: dict[str, list[str]] = {}

    for skill, pattern in _compiled_patterns().items():
        if pattern.search(normalized_text):
            category = skill_category(skill) or "Autres"
            found_by_category.setdefault(category, []).append(skill)

    for category in found_by_category:
        found_by_category[category].sort()

    return found_by_category


def flatten_skills(skills_by_category: dict[str, list[str]]) -> set[str]:
    """Flatten a {category: [skills]} dict into a single set of skill names."""
    flat: set[str] = set()
    for skills in skills_by_category.values():
        flat.update(skills)
    return flat


# --------------------------------------------------------------------------
# Years of experience
# --------------------------------------------------------------------------

_EXPERIENCE_PATTERNS = [
    # "5 ans d'expérience", "1 an d'experience", "5 années d'expérience"
    r"(\d{1,2})\s*\+?\s*(?:ans?|annees?)\s+d[’']?\s*exp[ée]rience",
    # "expérience de 5 ans" / "experience d'1 an"
    r"exp[ée]rience\s+(?:de\s+|d[’']\s*)?(\d{1,2})\s*(?:ans?|annees?)",
    # "5+ years of experience", "1 year experience"
    r"(\d{1,2})\s*\+?\s*years?\s+(?:of\s+)?experience",
    # "experience: 3 years", "experience of 5 years"
    r"experience\s*(?::|of)?\s*(\d{1,2})\s*\+?\s*years?",
    # "minimum 5 ans" / "minimum 1 an"
    r"minimum\s+(?:de\s+)?(\d{1,2})\s*(?:ans?|annees?)",
    # "minimum 5 years"
    r"minimum\s+(?:of\s+)?(\d{1,2})\s*\+?\s*years?",
]

_COMPILED_EXPERIENCE_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in _EXPERIENCE_PATTERNS]


def extract_experience_years(text: str) -> int | None:
    """
    Best-effort extraction of the number of years of experience mentioned.

    Returns the maximum value found (résumés often restate seniority in
    several places, e.g. summary + each role), or None if no pattern matches.

    Limitation: this is a regex heuristic, not a computation from actual
    employment date ranges. It works well for explicit statements
    ("5 ans d'expérience") but will miss résumés that only list job dates
    without stating a total explicitly.
    """
    normalized_text = normalize(text)
    values: list[int] = []
    for pattern in _COMPILED_EXPERIENCE_PATTERNS:
        for match in pattern.finditer(normalized_text):
            try:
                values.append(int(match.group(1)))
            except (ValueError, IndexError):
                continue
    return max(values) if values else None


# --------------------------------------------------------------------------
# Education level
# --------------------------------------------------------------------------

# Ordered low -> high; the highest matching level found wins.
_EDUCATION_LEVELS: list[tuple[int, str, list[str]]] = [
    (1, "Baccalauréat", [r"\bbac\b(?!\s*\+)", r"\bbaccalaureat\b", r"\bhigh school diploma\b"]),
    (2, "Bac+2 (BTS/DUT)", [r"bac\s*\+\s*2", r"\bbts\b", r"\bdut\b", r"\bassociate degree\b"]),
    (3, "Licence / Bachelor", [r"bac\s*\+\s*3", r"\blicence\b", r"\bbachelor\b", r"\bbachelor's degree\b"]),
    (4, "Master / Bac+5", [
        r"bac\s*\+\s*5", r"\bmaster\b", r"\bmastere\b", r"\bmba\b",
        r"diplome d[’']ingenieur", r"\bingenieur\b", r"\bmaster's degree\b",
    ]),
    (5, "Doctorat / PhD", [r"\bdoctorat\b", r"\bphd\b", r"\bph\.d\b", r"\bdoctorate\b"]),
]

_COMPILED_EDUCATION_LEVELS = [
    (level, label, [re.compile(p, flags=re.IGNORECASE) for p in patterns])
    for level, label, patterns in _EDUCATION_LEVELS
]


def extract_education_level(text: str) -> tuple[int, str] | None:
    """
    Detect the highest education level mentioned in the text.

    Returns (level: int 1-5, label: str) or None if nothing matched.
    """
    normalized_text = normalize(text)
    best: tuple[int, str] | None = None

    for level, label, patterns in _COMPILED_EDUCATION_LEVELS:
        if any(p.search(normalized_text) for p in patterns):
            if best is None or level > best[0]:
                best = (level, label)

    return best


# --------------------------------------------------------------------------
# Convenience container
# --------------------------------------------------------------------------


@dataclass
class ExtractedProfile:
    """Structured signals extracted from a single document (CV or job offer)."""

    raw_text: str
    skills_by_category: dict[str, list[str]] = field(default_factory=dict)
    skills_flat: set[str] = field(default_factory=set)
    experience_years: int | None = None
    education_level: tuple[int, str] | None = None

    @classmethod
    def from_text(cls, text: str) -> "ExtractedProfile":
        skills_by_category = find_skills(text)
        return cls(
            raw_text=text,
            skills_by_category=skills_by_category,
            skills_flat=flatten_skills(skills_by_category),
            experience_years=extract_experience_years(text),
            education_level=extract_education_level(text),
        )
