"""
tests/test_requirements_and_eligibility.py
-------------------------------------------
Unit tests for:
1. Missing mandatory (must-have) skill triggering hard-gate score capping at <= 60%.
2. Career duration calculation with overlapping tenure interval merging.
3. Neutral scoring (1.0) when experience or degree is omitted from the job description.
4. Academic degree tiered qualification scoring.
"""

import pytest
from src.core.models import (
    CandidateGraphPayload,
    Entity,
    EntityCategory,
    JobDescriptionGraphPayload,
    JobRequirement,
    PredicateType,
    Relation,
    merge_date_intervals_months,
)
from src.pipeline.ranker import HybridRanker


def test_missing_mandatory_skill_triggers_score_capping():
    """If a candidate misses an essential must-have skill, final score must be capped at 60.0%."""
    jd = JobDescriptionGraphPayload(
        job_id="job_must_test",
        title="ML Engineer",
        required_experience_years=2.0,
        required_degree_level=5,
        entities=[
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:pytorch", label="PyTorch", category=EntityCategory.SKILL),
        ],
        requirements=[
            JobRequirement(skill_id="skill:python", name="Python", importance="must"),
            JobRequirement(skill_id="skill:pytorch", name="PyTorch", importance="must"),
        ],
        raw_text="ML Engineer with Python and PyTorch mandatory.",
    )

    # Candidate has Python, but NO PyTorch
    cand = CandidateGraphPayload(
        candidate_id="cand_missing_must",
        name="Python Only Dev",
        degree_level=5,
        experience_years=5.0,
        entities=[
            Entity(id="cand_missing_must", label="Python Only Dev", category=EntityCategory.CANDIDATE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
        ],
        relations=[],
        raw_text="Developer with 5 years in Python.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand, jd)

    assert "PyTorch" in result.missing_must_requirements
    assert result.final_score <= 60.0
    assert result.is_capped_by_must_have


def test_overlapping_date_intervals_merging():
    """Overlapping tenures must be merged to prevent double-counting active career duration."""
    # Job A: Jan 2020 (2020*12 + 1) to Dec 2022 (2022*12 + 12) -> 36 months
    # Job B: Jan 2021 (2021*12 + 1) to Dec 2023 (2023*12 + 12) -> overlaps Job A
    # Total combined span: Jan 2020 to Dec 2023 -> 48 months (4.0 years), not 3 + 3 = 6 years!
    start_a = 2020 * 12 + 1
    end_a = 2022 * 12 + 12
    start_b = 2021 * 12 + 1
    end_b = 2023 * 12 + 12

    merged = merge_date_intervals_months([(start_a, end_a), (start_b, end_b)])
    assert len(merged) == 1
    assert merged[0][0] == start_a
    assert merged[0][1] == end_b
    total_months = merged[0][1] - merged[0][0]
    assert total_months == 47 or total_months == 48

    cand = CandidateGraphPayload(
        candidate_id="cand_overlap",
        name="Overlapping Roles Candidate",
        entities=[
            Entity(id="cand_overlap", label="Candidate", category=EntityCategory.CANDIDATE),
            Entity(id="role:dev1", label="Role 1", category=EntityCategory.ROLE),
            Entity(id="role:dev2", label="Role 2", category=EntityCategory.ROLE),
        ],
        relations=[
            Relation(subject_id="cand_overlap", predicate=PredicateType.HELD_ROLE, object_id="role:dev1", start_date="2020-01", end_date="2022-12"),
            Relation(subject_id="cand_overlap", predicate=PredicateType.HELD_ROLE, object_id="role:dev2", start_date="2021-01", end_date="2023-12"),
        ],
    )
    years = cand.compute_active_career_duration_years()
    # 48 months / 12 = 4.0 years (not 6 years!)
    assert pytest.approx(years, abs=0.2) == 4.0


def test_neutral_scoring_when_experience_or_degree_omitted():
    """If JD specifies no required experience or degree, sub-scores must default to 1.0 (neutral)."""
    jd_unconstrained = JobDescriptionGraphPayload(
        job_id="job_open",
        title="Software Contributor",
        required_experience_years=None,
        required_degree_level=None,
        entities=[
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
        ],
        raw_text="Open source Python contributor.",
    )

    cand = CandidateGraphPayload(
        candidate_id="cand_beginner",
        name="Self-taught Beginner",
        entities=[
            Entity(id="cand_beginner", label="Beginner", category=EntityCategory.CANDIDATE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
        ],
        relations=[],
        raw_text="Self-taught Python coder.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand, jd_unconstrained)

    assert result.experience_score == 1.0
    assert result.degree_score == 1.0


def test_degree_qualification_scoring_hierarchy():
    """Candidate with higher/equal degree gets 1.0; candidate with lower degree gets proportional score."""
    jd = JobDescriptionGraphPayload(
        job_id="job_master",
        title="Research Engineer",
        required_degree_level=5, # Bac+5
        entities=[Entity(id="skill:python", label="Python", category=EntityCategory.SKILL)],
        raw_text="Research Engineer with Master / Ingénieur.",
    )

    cand_phd = CandidateGraphPayload(
        candidate_id="cand_phd",
        name="PhD Holder",
        degree_level=8,
        entities=[Entity(id="cand_phd", label="PhD", category=EntityCategory.CANDIDATE)],
        raw_text="PhD in Computer Science.",
    )

    cand_bac2 = CandidateGraphPayload(
        candidate_id="cand_bac2",
        name="BTS Holder",
        degree_level=2,
        entities=[Entity(id="cand_bac2", label="BTS", category=EntityCategory.CANDIDATE)],
        raw_text="BTS Informatique (Bac+2).",
    )

    ranker = HybridRanker()
    res_phd = ranker.score_candidate(cand_phd, jd)
    res_bac2 = ranker.score_candidate(cand_bac2, jd)

    assert res_phd.degree_score == 1.0
    assert res_bac2.degree_score == pytest.approx(2.0 / 5.0, abs=1e-2)
    assert res_phd.degree_score > res_bac2.degree_score
