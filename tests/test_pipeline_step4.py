"""
test_pipeline_step4.py
------------------------
Step 4 tests for the generate_explanation node.

- Mocked tests (default, run in CI / no API key needed): verify the LLM
  path is used when available, that the fallback template kicks in on a
  failure or empty response, and that the full graph exposes "explanation"
  in its final state.
- One live test (skipped by default): actually calls Groq, run manually
  with `RUN_LIVE_LLM_TEST=1 pytest tests/test_pipeline_step4.py -v -s`
  to sanity-check the real integration end to end.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.extractor import ExtractedProfile
from src.scoring.scorer import score_candidate
from src.graph.nodes import generate_explanation, _fallback_explanation
from src.graph.pipeline import graph

JOB_TEXT = """
Nous recherchons un Développeur Backend Python (H/F).
Profil recherché : Bac+5, 5 ans d'expérience minimum en développement backend.
Maîtrise de Python, Django, PostgreSQL.
"""

STRONG_CV = """
Karim Ben Ali - Ingénieur Logiciel
Master en Informatique.
7 ans d'expérience en développement backend Python.
Compétences techniques : Python, Django, PostgreSQL, Docker, AWS.
"""

WEAK_CV = """
Sami Trabelsi - Développeur Web Junior
Licence en informatique.
1 an d'expérience en développement web avec PHP et MySQL.
"""


def _make_fake_llm_response(text: str):
    fake_response = MagicMock()
    fake_response.content = text
    return fake_response


@patch("src.graph.nodes._llm")
def test_generate_explanation_uses_llm_output(mock_llm):
    mock_llm.invoke.return_value = _make_fake_llm_response(
        "Ce candidat dispose d'une solide expérience en développement backend "
        "Python et maîtrise les technologies clés demandées par le poste. "
        "Son profil est à retenir en priorité."
    )

    breakdown = score_candidate(
        ExtractedProfile.from_text(STRONG_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    result = generate_explanation({"score_breakdown": breakdown})

    assert "retenir en priorité" in result["explanation"]
    mock_llm.invoke.assert_called_once()


@patch("src.graph.nodes._llm")
def test_generate_explanation_strips_wrapping_quotes(mock_llm):
    mock_llm.invoke.return_value = _make_fake_llm_response(
        '"Ce candidat présente un profil junior avec une expérience limitée."'
    )

    breakdown = score_candidate(
        ExtractedProfile.from_text(WEAK_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    result = generate_explanation({"score_breakdown": breakdown})

    assert not result["explanation"].startswith('"')
    assert not result["explanation"].endswith('"')


@patch("src.graph.nodes._llm")
def test_generate_explanation_falls_back_on_empty_response(mock_llm):
    mock_llm.invoke.return_value = _make_fake_llm_response("   ")

    breakdown = score_candidate(
        ExtractedProfile.from_text(WEAK_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    result = generate_explanation({"score_breakdown": breakdown})

    assert result["explanation"] == _fallback_explanation(breakdown)


@patch("src.graph.nodes._llm")
def test_generate_explanation_falls_back_on_api_exception(mock_llm):
    mock_llm.invoke.side_effect = Exception("API timeout")

    breakdown = score_candidate(
        ExtractedProfile.from_text(STRONG_CV),
        ExtractedProfile.from_text(JOB_TEXT),
    )
    result = generate_explanation({"score_breakdown": breakdown})

    assert result["explanation"] == _fallback_explanation(breakdown)


def test_fallback_explanation_is_never_empty_even_with_no_signals():
    # A breakdown with no strengths/weaknesses at all should still produce
    # a non-empty opening sentence, not an empty string.
    breakdown = score_candidate(
        ExtractedProfile.from_text("Texte sans aucun signal détectable."),
        ExtractedProfile.from_text("Texte sans aucun signal détectable non plus."),
    )
    explanation = _fallback_explanation(breakdown)
    assert explanation.strip() != ""


@patch("src.graph.nodes._llm")
def test_full_graph_exposes_explanation(mock_llm):
    mock_llm.invoke.return_value = _make_fake_llm_response(
        "Synthèse générée pour test d'intégration du graphe complet."
    )

    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": STRONG_CV})

    assert result["score_breakdown"] is not None
    assert result.get("explanation")


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TEST") != "1",
    reason="Live LLM test skipped by default — set RUN_LIVE_LLM_TEST=1 to run it against the real Groq API.",
)
def test_live_groq_explanation_end_to_end():
    """Manual sanity check against the real Groq API. Requires GROQ_API_KEY."""
    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": STRONG_CV})
    print("\nLive explanation:", result["explanation"])
    assert result["explanation"]


def run_all():
    tests = [
        test_fallback_explanation_is_never_empty_even_with_no_signals,
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
    print("\nTous les tests directement exécutables sont passés (lancez pytest pour les tests mockés).")


if __name__ == "__main__":
    run_all()