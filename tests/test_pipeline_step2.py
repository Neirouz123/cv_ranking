"""
test_pipeline_step2.py
------------------------
Step 2 regression test: verifies the confidence-check conditional edge
routes correctly.

- WEAK_CV_INCOMPLETE (missing education) should trigger llm_extract_cv (stub).
- STRONG_CV (has skills, experience, education) should skip straight to
  compute_score, never touching llm_extract_cv.

Run directly: python3 tests/test_pipeline_step2.py
Or via pytest: pytest tests/test_pipeline_step2.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.nodes import check_extraction_confidence
from src.extraction.extractor import ExtractedProfile
from src.graph.pipeline import graph

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

# Deliberately vague CV: has skills, but no explicit years-of-experience
# phrase and no education level mentioned -> should trigger escalation.
VAGUE_CV = """
Amine Cherif - Développeur
Je code en Python et JavaScript depuis plusieurs années, j'ai travaillé
avec Django et React sur plusieurs projets d'entreprise.
"""


def test_confidence_check_escalates_on_vague_cv():
    profile = ExtractedProfile.from_text(VAGUE_CV)
    decision = check_extraction_confidence({"cv_profile": profile})
    assert decision == "llm_extract_cv"


def test_confidence_check_skips_llm_on_strong_cv():
    profile = ExtractedProfile.from_text(STRONG_CV)
    decision = check_extraction_confidence({"cv_profile": profile})
    assert decision == "compute_score"


def test_graph_runs_end_to_end_with_vague_cv(capsys):
    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": VAGUE_CV})
    captured = capsys.readouterr()

    assert result["score_breakdown"] is not None
    assert "llm_extract_cv" in captured.out  # confirms escalation actually happened


def test_graph_skips_stub_with_strong_cv(capsys):
    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": STRONG_CV})
    captured = capsys.readouterr()

    assert result["score_breakdown"] is not None
    assert "llm_extract_cv" not in captured.out  # confirms it was skipped


def run_all():
    tests = [
        test_confidence_check_escalates_on_vague_cv,
        test_confidence_check_skips_llm_on_strong_cv,
    ]
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
    print("\nTous les tests sont passés (routing only — run via pytest for stdout-capture tests).")


if __name__ == "__main__":
    run_all()