"""
tests/test_bilingual_tfidf.py
------------------------------
Unit tests for CorpusVectorizer:
- Verifies that French and English stopwords do not inflate similarity.
- Verifies diacritic normalization and punctuation cleaning.
- Verifies batch similarity computation against ground truth documents.
"""

import pytest
from src.pipeline.vector_store import CorpusVectorizer, sanitize_text


def test_sanitize_text_normalizes_diacritics_and_noise():
    raw = "Ingénieur Logiciel Sénior, email: contact@domain.com, https://github.com/profile, 2024!"
    cleaned = sanitize_text(raw)
    assert "ingenieur" in cleaned
    assert "logiciel" in cleaned
    assert "senior" in cleaned
    assert "contact@domain.com" not in cleaned
    assert "https" not in cleaned
    assert "2024" not in cleaned


def test_french_stopwords_do_not_inflate_similarity():
    """Two documents sharing only French stopwords must have near-zero or zero cosine similarity."""
    text_a = "Dans le cadre de un projet pour le compte de une organisation."
    text_b = "Sur le sujet avec des propositions dans la structure de travail."

    vectorizer = CorpusVectorizer()
    sim = vectorizer.compute_similarity(text_a, text_b)
    # Stopwords are removed; remaining unique content words have no overlap
    assert sim == pytest.approx(0.0, abs=1e-3)


def test_meaningful_technical_terms_have_high_similarity():
    """Matching technical terms in bilingual contexts should yield solid similarity."""
    jd = "Recherche Ingénieur Machine Learning avec expertise Python, PyTorch et Scikit-learn."
    cv_match = "Machine Learning Engineer with solid experience in Python and PyTorch deep learning models."
    cv_unrelated = "Chef de cuisine gastronomique gestion brigade hôtellerie."

    vectorizer = CorpusVectorizer()
    vectorizer.fit([jd, cv_match, cv_unrelated])

    sim_match = vectorizer.compute_similarity(jd, cv_match)
    sim_unrelated = vectorizer.compute_similarity(jd, cv_unrelated)

    assert sim_match > 0.30
    assert sim_match > sim_unrelated
    assert sim_unrelated == pytest.approx(0.0, abs=0.05)


def test_batch_similarities():
    jd = "Développeur Fullstack React FastAPI PostgreSQL Docker"
    candidates = [
        "Fullstack Developer proficient in React, FastAPI and PostgreSQL.",
        "Comptable gestion de paie clôture bilan fiscal.",
    ]
    vectorizer = CorpusVectorizer()
    sims = vectorizer.compute_batch_similarities(jd, candidates)
    assert len(sims) == 2
    assert sims[0] > 0.35
    assert sims[1] < 0.05
    assert sims[0] > sims[1]
