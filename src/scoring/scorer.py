"""
scorer.py
---------
Weighted scoring algorithm that ranks a candidate's résumé against a job
offer, combining four independently-computed, explainable components:

1. Skills match   - proportion of job-required skills found in the résumé
2. Text relevance - TF-IDF cosine similarity between full résumé & offer text
                     (catches domain vocabulary not in the curated skills list)
3. Experience fit  - candidate's years of experience vs. years required
4. Education fit   - candidate's education level vs. level required

Each component is scored 0-100, then combined via user-adjustable weights
(must sum to 1.0). The design goal is transparency: every subscore and every
"weakness" statement is derived from a traceable rule, not a black box -
suitable for an HR process that may need to justify a ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..extraction.extractor import ExtractedProfile

# French + English stopwords kept minimal and merged, since job offers/CVs
# in this tool are bilingual and scikit-learn's built-in list is English-only.
_EXTRA_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "dans",
    "pour", "sur", "avec", "au", "aux", "ce", "ces", "cette", "est", "sont",
    "vous", "nous", "notre", "vos", "votre", "il", "elle", "ils", "elles",
    "que", "qui", "ne", "pas", "plus", "ou", "où", "par", "son", "sa", "ses",
    "nos", "leur", "leurs", "afin", "ainsi", "avoir", "être", "vers",
}


@dataclass
class ScoringWeights:
    """Relative importance of each scoring component. Must sum to 1.0."""

    skills: float = 0.50
    text_relevance: float = 0.20
    experience: float = 0.15
    education: float = 0.15

    def normalized(self) -> "ScoringWeights":
        total = self.skills + self.text_relevance + self.experience + self.education
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        return ScoringWeights(
            skills=self.skills / total,
            text_relevance=self.text_relevance / total,
            experience=self.experience / total,
            education=self.education / total,
        )


@dataclass
class ScoreBreakdown:
    """Full, explainable result of scoring one candidate against one offer."""

    overall_score: float

    skills_score: float
    text_relevance_score: float
    experience_score: float
    education_score: float

    matched_skills: dict[str, list[str]]
    missing_skills: dict[str, list[str]]

    candidate_experience_years: int | None
    required_experience_years: int | None

    candidate_education: tuple[int, str] | None
    required_education: tuple[int, str] | None

    weaknesses: list[str]
    strengths: list[str]


def _text_relevance_score(cv_text: str, job_text: str) -> float:
    """TF-IDF cosine similarity between the two documents, scaled to 0-100."""
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(_EXTRA_STOPWORDS) or None,
            max_features=2000,
        )
        matrix = vectorizer.fit_transform([job_text, cv_text])
        similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
        return round(float(similarity) * 100, 1)
    except ValueError:
        # Can happen on extremely short/empty documents after stopword removal
        return 0.0


def _experience_score(candidate_years: int | None, required_years: int | None) -> float:
    """
    100 if candidate meets or exceeds the requirement.
    Linear partial credit if below. Neutral (70) if requirement is unknown
    but candidate states some experience; 50 if nothing could be determined
    on either side (avoids unfairly zeroing out résumés with no explicit
    "X years" statement).
    """
    if required_years is None:
        return 70.0 if candidate_years else 50.0
    if candidate_years is None:
        return 30.0
    if candidate_years >= required_years:
        return 100.0
    return round(max(0.0, candidate_years / required_years) * 100, 1)


def _education_score(
    candidate_level: tuple[int, str] | None, required_level: tuple[int, str] | None
) -> float:
    """Same logic as experience: meets/exceeds -> 100, below -> partial credit."""
    if required_level is None:
        return 70.0 if candidate_level else 50.0
    if candidate_level is None:
        return 30.0
    cand, req = candidate_level[0], required_level[0]
    if cand >= req:
        return 100.0
    return round(max(0.0, cand / req) * 100, 1)


def _build_weaknesses_and_strengths(
    missing_skills: dict[str, list[str]],
    matched_skills: dict[str, list[str]],
    candidate_years: int | None,
    required_years: int | None,
    candidate_edu: tuple[int, str] | None,
    required_edu: tuple[int, str] | None,
) -> tuple[list[str], list[str]]:
    weaknesses: list[str] = []
    strengths: list[str] = []

    flat_missing = [s for skills in missing_skills.values() for s in skills]
    flat_matched = [s for skills in matched_skills.values() for s in skills]

    if flat_missing:
        shown = ", ".join(flat_missing[:6])
        suffix = "…" if len(flat_missing) > 6 else ""
        weaknesses.append(f"Compétences requises absentes du CV : {shown}{suffix}")

    if flat_matched:
        shown = ", ".join(flat_matched[:6])
        suffix = "…" if len(flat_matched) > 6 else ""
        strengths.append(f"Compétences correspondant à l'offre : {shown}{suffix}")

    if required_years is not None:
        if candidate_years is None:
            weaknesses.append(
                f"Le CV ne précise pas le nombre d'années d'expérience "
                f"(l'offre en demande {required_years})."
            )
        elif candidate_years < required_years:
            weaknesses.append(
                f"Expérience en dessous du seuil demandé : {candidate_years} an(s) "
                f"contre {required_years} requis."
            )
        else:
            strengths.append(
                f"Expérience suffisante : {candidate_years} an(s) pour {required_years} requis."
            )

    if required_edu is not None:
        if candidate_edu is None:
            weaknesses.append(
                f"Le niveau de formation n'est pas identifiable dans le CV "
                f"(l'offre demande : {required_edu[1]})."
            )
        elif candidate_edu[0] < required_edu[0]:
            weaknesses.append(
                f"Niveau de formation en dessous de l'attendu : {candidate_edu[1]} "
                f"contre {required_edu[1]} requis."
            )
        else:
            strengths.append(f"Niveau de formation adéquat : {candidate_edu[1]}.")

    if not flat_matched and not flat_missing:
        weaknesses.append(
            "Aucune compétence de la taxonomie n'a été détectée dans l'offre ou le CV : "
            "le score de compétences n'est pas discriminant ici, fiez-vous davantage "
            "à la pertinence textuelle."
        )

    return weaknesses, strengths


def score_candidate(
    cv_profile: ExtractedProfile,
    job_profile: ExtractedProfile,
    weights: ScoringWeights | None = None,
) -> ScoreBreakdown:
    """
    Compute the full weighted score of a candidate's résumé against a job offer.

    Args:
        cv_profile: ExtractedProfile built from the candidate's résumé text.
        job_profile: ExtractedProfile built from the job offer text.
        weights: relative importance of each component (auto-normalized).

    Returns:
        ScoreBreakdown with the overall 0-100 score and full component detail.
    """
    weights = (weights or ScoringWeights()).normalized()

    required_skills = job_profile.skills_flat
    candidate_skills = cv_profile.skills_flat

    matched = required_skills & candidate_skills
    missing = required_skills - candidate_skills

    matched_by_cat = {
        cat: [s for s in skills if s in matched]
        for cat, skills in job_profile.skills_by_category.items()
    }
    matched_by_cat = {cat: skills for cat, skills in matched_by_cat.items() if skills}

    missing_by_cat = {
        cat: [s for s in skills if s in missing]
        for cat, skills in job_profile.skills_by_category.items()
    }
    missing_by_cat = {cat: skills for cat, skills in missing_by_cat.items() if skills}

    skills_score = round((len(matched) / len(required_skills)) * 100, 1) if required_skills else 50.0
    text_score = _text_relevance_score(cv_profile.raw_text, job_profile.raw_text)
    experience_score = _experience_score(cv_profile.experience_years, job_profile.experience_years)
    education_score = _education_score(cv_profile.education_level, job_profile.education_level)

    overall = (
        skills_score * weights.skills
        + text_score * weights.text_relevance
        + experience_score * weights.experience
        + education_score * weights.education
    )

    weaknesses, strengths = _build_weaknesses_and_strengths(
        missing_by_cat,
        matched_by_cat,
        cv_profile.experience_years,
        job_profile.experience_years,
        cv_profile.education_level,
        job_profile.education_level,
    )

    return ScoreBreakdown(
        overall_score=round(overall, 1),
        skills_score=skills_score,
        text_relevance_score=text_score,
        experience_score=experience_score,
        education_score=education_score,
        matched_skills=matched_by_cat,
        missing_skills=missing_by_cat,
        candidate_experience_years=cv_profile.experience_years,
        required_experience_years=job_profile.experience_years,
        candidate_education=cv_profile.education_level,
        required_education=job_profile.education_level,
        weaknesses=weaknesses,
        strengths=strengths,
    )
