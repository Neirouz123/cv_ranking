"""
test_pipeline_step1.py
------------------------
Step 1 regression test for the LangGraph pipeline: verifies that running
job/CV extraction + scoring through the graph (src/graph/pipeline.py)
produces the exact same ScoreBreakdown as calling extractor.py / scorer.py
directly (the pre-LangGraph approach used in test_pipeline.py).

Run directly: python3 tests/test_pipeline_step1.py
Or via pytest: pytest tests/test_pipeline_step1.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.extractor import ExtractedProfile
from src.scoring.scorer import score_candidate
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

WEAK_CV = """
Sami Trabelsi - Développeur Web Junior
Licence en informatique.
1 an d'expérience en développement web avec PHP et MySQL.
Connaissances de base en JavaScript et HTML/CSS.
"""


def test_graph_matches_direct_call_strong_cv():
    direct_result = score_candidate(
        ExtractedProfile.from_text(STRONG_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    graph_result = graph.invoke({"job_text": JOB_TEXT, "cv_text": STRONG_CV})

    assert graph_result["score_breakdown"] == direct_result


def test_graph_matches_direct_call_weak_cv():
    direct_result = score_candidate(
        ExtractedProfile.from_text(WEAK_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    graph_result = graph.invoke({"job_text": JOB_TEXT, "cv_text": WEAK_CV})

    assert graph_result["score_breakdown"] == direct_result


def test_graph_state_contains_profiles():
    # Sanity check that intermediate nodes actually populated the state,
    # not just the final score.
    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": STRONG_CV})
    assert result["job_profile"] is not None
    assert result["cv_profile"] is not None
    assert "Python" in result["cv_profile"].skills_flat


def run_all():
    tests = [
        test_graph_matches_direct_call_strong_cv,
        test_graph_matches_direct_call_weak_cv,
        test_graph_state_contains_profiles,
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
    print("\nTous les tests sont passés.")


if __name__ == "__main__":
    run_all()