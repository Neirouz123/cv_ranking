"""
test_pipeline_step3.py
------------------------
Step 3 tests for the real Groq-based llm_extract_cv node.

- Mocked tests (default, run in CI / no API key needed): verify JSON
  parsing, taxonomy mapping, and fallback-on-failure behavior without
  hitting the real API.
- One live test (skipped by default): actually calls Groq, run manually
  with `RUN_LIVE_LLM_TEST=1 pytest tests/test_pipeline_step3.py -v -s`
  to sanity-check the real integration end to end.
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.extractor import ExtractedProfile
from src.graph.nodes import llm_extract_cv
from src.graph.pipeline import graph

JOB_TEXT = """
Nous recherchons un Développeur Backend Python (H/F).
Profil recherché : Bac+5, 5 ans d'expérience minimum en développement backend.
Maîtrise de Python, Django, PostgreSQL.
"""

VAGUE_CV_TEXT = """
Amine Cherif - Développeur
Je code en Python et JavaScript depuis plusieurs années, j'ai travaillé
avec Django et React sur plusieurs projets d'entreprise.
"""


def _make_fake_llm_response(json_payload: dict):
    fake_response = MagicMock()
    fake_response.content = json.dumps(json_payload)
    return fake_response


@patch("src.graph.nodes._llm")
def test_llm_extract_cv_parses_valid_json(mock_llm):
    mock_llm.invoke.return_value = _make_fake_llm_response({
        "skills": ["Python", "Django", "React"],
        "experience_years": 4,
        "education_level": 4,
    })

    fallback_profile = ExtractedProfile.from_text(VAGUE_CV_TEXT)
    state = {"cv_text": VAGUE_CV_TEXT, "cv_profile": fallback_profile}

    result = llm_extract_cv(state)
    profile = result["cv_profile"]

    assert profile.experience_years == 4
    assert profile.education_level == (4, "Master / Bac+5")
    assert "Python" in profile.skills_flat


@patch("src.graph.nodes._llm")
def test_llm_extract_cv_handles_markdown_fences(mock_llm):
    # Some models wrap JSON in ```json ... ``` despite instructions not to
    payload = json.dumps({"skills": ["Python"], "experience_years": 2, "education_level": 3})
    mock_llm.invoke.return_value = _make_fake_llm_response({})  # placeholder, overwritten below
    mock_llm.invoke.return_value.content = f"```json\n{payload}\n```"

    fallback_profile = ExtractedProfile.from_text(VAGUE_CV_TEXT)
    state = {"cv_text": VAGUE_CV_TEXT, "cv_profile": fallback_profile}

    result = llm_extract_cv(state)
    assert result["cv_profile"].experience_years == 2


@patch("src.graph.nodes._llm")
def test_llm_extract_cv_falls_back_on_malformed_json(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="this is not json at all")

    fallback_profile = ExtractedProfile.from_text(VAGUE_CV_TEXT)
    state = {"cv_text": VAGUE_CV_TEXT, "cv_profile": fallback_profile}

    result = llm_extract_cv(state)
    # Should fall back to the regex-based profile unchanged
    assert result["cv_profile"] == fallback_profile


@patch("src.graph.nodes._llm")
def test_llm_extract_cv_falls_back_on_api_exception(mock_llm):
    mock_llm.invoke.side_effect = Exception("API timeout")

    fallback_profile = ExtractedProfile.from_text(VAGUE_CV_TEXT)
    state = {"cv_text": VAGUE_CV_TEXT, "cv_profile": fallback_profile}

    result = llm_extract_cv(state)
    assert result["cv_profile"] == fallback_profile


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TEST") != "1",
    reason="Live LLM test skipped by default — set RUN_LIVE_LLM_TEST=1 to run it against the real Groq API.",
)
def test_live_groq_extraction_end_to_end():
    """Manual sanity check against the real Groq API. Requires GROQ_API_KEY."""
    result = graph.invoke({"job_text": JOB_TEXT, "cv_text": VAGUE_CV_TEXT})
    print("\nLive extraction result:", result["cv_profile"])
    print("Score breakdown:", result["score_breakdown"])
    assert result["score_breakdown"] is not None