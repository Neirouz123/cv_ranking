"""
tests/test_scoring_parity.py
-----------------------------
Unit tests verifying that:
1. HybridRanker.score_candidate and compute_final_rank return identical results for identical inputs.
2. An ideal profile without conflicts achieves exactly 100.0 (no cap at 90).
3. Zero positive weights edge case falls back safely to 0.0.
"""

import pytest
from src.core.models import (
    CandidateGraphPayload,
    Entity,
    EntityCategory,
    JobDescriptionGraphPayload,
    PredicateType,
    Relation,
)
from src.pipeline.ranker import HybridRanker, RankingWeights, compute_final_rank


def test_ideal_candidate_achieves_perfect_100():
    """An ideal candidate fulfilling all requirements with 0 conflicts must score exactly 100.0."""
    jd = JobDescriptionGraphPayload(
        job_id="job_perfect",
        title="Fullstack Engineer",
        required_experience_years=2.0,
        required_degree_level=5,
        entities=[
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:react", label="React", category=EntityCategory.SKILL),
        ],
        raw_text="Fullstack Engineer with Python and React.",
    )

    cand = CandidateGraphPayload(
        candidate_id="cand_ideal",
        name="Perfect Match",
        degree_level=5,
        experience_years=3.0,
        entities=[
            Entity(id="cand_ideal", label="Perfect Match", category=EntityCategory.CANDIDATE),
            Entity(id="role:dev", label="Developer", category=EntityCategory.ROLE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:react", label="React", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(
                subject_id="cand_ideal",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:dev",
                start_date="2021-01-01",
                end_date="2024-01-01",
            ),
            Relation(
                subject_id="role:dev",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:python",
                start_date="2021-01-01",
                end_date="2024-01-01",
            ),
            Relation(
                subject_id="role:dev",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:react",
                start_date="2021-01-01",
                end_date="2024-01-01",
            ),
        ],
        raw_text="Fullstack Engineer with Python and React.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand, jd)

    # Must NOT be capped at 90; must reach 100.0
    assert result.final_score == 100.0
    assert result.conflict_penalty == 0.0
    assert not result.is_capped_by_must_have


def test_parity_between_ranker_and_compute_final_rank():
    """HybridRanker.score_candidate and compute_final_rank must yield identical numbers."""
    jd = JobDescriptionGraphPayload(
        job_id="job_parity",
        title="Python Dev",
        required_experience_years=3.0,
        required_degree_level=3,
        entities=[
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
        ],
        raw_text="Python Developer needed with 3 years experience.",
    )

    cand = CandidateGraphPayload(
        candidate_id="cand_parity",
        name="Parity Candidate",
        degree_level=3,
        experience_years=3.0,
        entities=[
            Entity(id="cand_parity", label="Parity Candidate", category=EntityCategory.CANDIDATE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
        ],
        relations=[],
        raw_text="Python Developer with strong Python skills.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand, jd)

    # Compute directly via compute_final_rank
    direct_score = compute_final_rank(
        vector_score=result.vector_similarity,
        graph_score=result.graph_match_score,
        penalty_score=result.conflict_penalty,
        exp_score=result.experience_score,
        degree_score=result.degree_score,
        weights=ranker.weights,
        has_missing_must=bool(result.missing_must_requirements),
        scale_100=True,
    )

    assert result.final_score == pytest.approx(direct_score, abs=1e-2)


def test_zero_positive_weights_safely_falls_back():
    """When all positive weights are 0.0, compute_final_rank must return 0.0 without dividing by zero."""
    zero_weights = RankingWeights(alpha=0.0, beta=0.0, delta=0.0, epsilon=0.0, gamma=0.25)
    score = compute_final_rank(
        vector_score=1.0,
        graph_score=1.0,
        penalty_score=0.0,
        exp_score=1.0,
        degree_score=1.0,
        weights=zero_weights,
        scale_100=True,
    )
    assert score == 0.0
