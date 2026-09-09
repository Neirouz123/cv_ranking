"""
test_pipeline.py
-----------------
Lightweight end-to-end sanity check of extraction + scoring, with no external
test framework dependency (run directly with `python3 tests/test_pipeline.py`).

For a real project, port these into pytest test functions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.extractor import ExtractedProfile, extract_experience_years, extract_education_level
from src.scoring.scorer import ScoringWeights, score_candidate

JOB_TEXT = """
Nous recherchons un Développeur Backend Python (H/F).
Missions : conception d'API avec Django et FastAPI, travail avec PostgreSQL et Docker,
déploiement sur AWS, mise en place de pipelines CI/CD.
Profil recherché : Bac+5 (Master ou diplôme d'ingénieur), 5 ans d'expérience minimum
en développement backend. Maîtrise de Git, méthodologie Agile/Scrum.
Anglais professionnel requis.
"""

STRONG_CV = """
Karim Ben Ali - Ingénieur Logiciel
Master en Informatique, Ecole Nationale d'Ingénieurs de Tunis.
7 ans d'expérience en développement backend Python.
Compétences techniques : Python, Django, FastAPI, PostgreSQL, Docker, AWS, Git, CI/CD.
Méthodologie Agile/Scrum au quotidien. Anglais professionnel courant.
"""

WEAK_CV = """
Sami Trabelsi - Développeur Web Junior
Licence en informatique.
1 an d'expérience en développement web avec PHP et MySQL.
Connaissances de base en JavaScript et HTML/CSS.
"""


def test_experience_extraction():
    assert extract_experience_years("1 an d'expérience") == 1
    assert extract_experience_years("7 ans d'expérience") == 7
    assert extract_experience_years("5+ years of experience") == 5
    assert extract_experience_years("aucune mention") is None


def test_education_extraction():
    assert extract_education_level("Master en informatique") == (4, "Master / Bac+5")
    assert extract_education_level("PhD in Computer Science") == (5, "Doctorat / PhD")
    assert extract_education_level("sans diplôme mentionné ici") is None


def test_scoring_discriminates_strong_vs_weak():
    job_profile = ExtractedProfile.from_text(JOB_TEXT)
    strong = ExtractedProfile.from_text(STRONG_CV)
    weak = ExtractedProfile.from_text(WEAK_CV)

    weights = ScoringWeights()
    strong_result = score_candidate(strong, job_profile, weights)
    weak_result = score_candidate(weak, job_profile, weights)

    assert strong_result.overall_score > weak_result.overall_score
    assert strong_result.overall_score >= 80
    assert weak_result.overall_score <= 40
    assert "Python" in {s for skills in strong_result.matched_skills.values() for s in skills}
    assert len(weak_result.weaknesses) > 0


def run_all():
    tests = [test_experience_extraction, test_education_extraction, test_scoring_discriminates_strong_vs_weak]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"❌ {test.__name__} — {exc}")
    if failures:
        print(f"\n{failures} test(s) échoué(s).")
        sys.exit(1)
    print("\nTous les tests sont passés.")


if __name__ == "__main__":
    run_all()
