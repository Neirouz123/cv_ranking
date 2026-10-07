"""
tests/test_contradictions.py
----------------------------
Unit tests for the consistency and contradiction detection engine:
- Temporal inversions (start_date > end_date)
- Cumulative over-claiming (claimed years > documented experience)
- Anachronisms (using technology before its release date)
- Clean candidate verification
"""

import pytest
from src.core.models import (
    CandidateGraphPayload,
    ConflictSeverity,
    ConflictType,
    Entity,
    EntityCategory,
    PredicateType,
    Relation,
)
from src.core.validator import ConsistencyValidator


@pytest.fixture
def validator():
    return ConsistencyValidator()


def test_clean_candidate_passes_validation(validator):
    """A consistent candidate with realistic dates should have 0 conflicts and 0 penalty."""
    payload = CandidateGraphPayload(
        candidate_id="cand_clean",
        name="Marie Curie",
        entities=[
            Entity(id="cand_clean", label="Marie Curie", category=EntityCategory.CANDIDATE),
            Entity(id="role:ml_engineer", label="ML Engineer", category=EntityCategory.ROLE),
            Entity(id="company:techcorp", label="TechCorp", category=EntityCategory.COMPANY),
            Entity(id="skill:pytorch", label="PyTorch", category=EntityCategory.SKILL, claimed_years=2.0),
        ],
        relations=[
            Relation(
                subject_id="cand_clean",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:ml_engineer",
                start_date="2020-01-01",
                end_date="2023-01-01",
            ),
            Relation(
                subject_id="role:ml_engineer",
                predicate=PredicateType.WORKED_AT,
                object_id="company:techcorp",
                start_date="2020-01-01",
                end_date="2023-01-01",
            ),
            Relation(
                subject_id="role:ml_engineer",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:pytorch",
                start_date="2020-01-01",
                end_date="2023-01-01",
            ),
        ],
        raw_text="Marie Curie, ML Engineer at TechCorp from 2020 to 2023 using PyTorch.",
    )

    report = validator.validate(payload)
    assert report.is_valid is True
    assert report.penalty_factor == 0.0
    assert len(report.conflicts) == 0


def test_temporal_inversion_detected(validator):
    """Detects start_date > end_date on a role or project."""
    payload = CandidateGraphPayload(
        candidate_id="cand_inversion",
        name="Temporal Traveler",
        entities=[
            Entity(id="cand_inversion", label="Temporal Traveler", category=EntityCategory.CANDIDATE),
            Entity(id="role:dev", label="Software Developer", category=EntityCategory.ROLE),
        ],
        relations=[
            Relation(
                subject_id="cand_inversion",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:dev",
                start_date="2023-01-01",
                end_date="2020-01-01",  # Inversion: start > end
            )
        ],
    )

    report = validator.validate(payload)
    assert report.penalty_factor > 0.0
    inversion_conflicts = [
        c for c in report.conflicts if c.conflict_type == ConflictType.TEMPORAL_INVERSION
    ]
    assert len(inversion_conflicts) == 1
    assert inversion_conflicts[0].severity == ConflictSeverity.HIGH
    assert "start_date" in inversion_conflicts[0].message


def test_anachronism_detected(validator):
    """Detects claiming technology usage prior to its official release year."""
    # FastAPI was released in 2018; candidate claims usage in 2014
    payload = CandidateGraphPayload(
        candidate_id="cand_anachronism",
        name="Early Adopter",
        entities=[
            Entity(id="cand_anachronism", label="Early Adopter", category=EntityCategory.CANDIDATE),
            Entity(id="role:webdev", label="Web Developer", category=EntityCategory.ROLE),
            Entity(id="skill:fastapi", label="FastAPI", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(
                subject_id="cand_anachronism",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:webdev",
                start_date="2014-01-01",
                end_date="2016-01-01",
            ),
            Relation(
                subject_id="role:webdev",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:fastapi",
                start_date="2014-01-01",
                end_date="2016-01-01",
            ),
        ],
    )

    report = validator.validate(payload)
    anachronisms = [
        c for c in report.conflicts if c.conflict_type == ConflictType.ANACHRONISM
    ]
    assert len(anachronisms) >= 1
    assert anachronisms[0].severity == ConflictSeverity.HIGH
    assert report.penalty_factor >= 0.35
    assert "2018" in anachronisms[0].message
    assert "2014" in anachronisms[0].message


def test_cumulative_overclaiming_detected(validator):
    """Flags candidate claiming N years when documented roles total < N."""
    # Claims 8 years of Python experience, but only has 1 year of documented role
    payload = CandidateGraphPayload(
        candidate_id="cand_overclaim",
        name="Exaggerating Dev",
        entities=[
            Entity(id="cand_overclaim", label="Exaggerating Dev", category=EntityCategory.CANDIDATE),
            Entity(id="role:junior_dev", label="Junior Developer", category=EntityCategory.ROLE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL, claimed_years=8.0),
        ],
        relations=[
            Relation(
                subject_id="cand_overclaim",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:junior_dev",
                start_date="2022-01-01",
                end_date="2023-01-01",  # 1 year
            ),
            Relation(
                subject_id="role:junior_dev",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:python",
                start_date="2022-01-01",
                end_date="2023-01-01",
            ),
        ],
    )

    report = validator.validate(payload)
    overclaims = [
        c for c in report.conflicts if c.conflict_type == ConflictType.CUMULATIVE_OVER_CLAIMING
    ]
    assert len(overclaims) == 1
    assert overclaims[0].severity == ConflictSeverity.MEDIUM
    assert report.penalty_factor >= 0.25
    assert "8.0" in overclaims[0].message


def test_multiple_contradictions_accumulate_penalty(validator):
    """Multiple conflicts should accumulate penalties up to the configured ceiling."""
    payload = CandidateGraphPayload(
        candidate_id="cand_multi_conflict",
        name="Multi Conflict",
        entities=[
            Entity(id="cand_multi_conflict", label="Multi Conflict", category=EntityCategory.CANDIDATE),
            Entity(id="role:bad_role", label="Role", category=EntityCategory.ROLE),
            Entity(id="skill:fastapi", label="FastAPI", category=EntityCategory.SKILL, claimed_years=10.0),
        ],
        relations=[
            # Temporal inversion:
            Relation(
                subject_id="cand_multi_conflict",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:bad_role",
                start_date="2024-01-01",
                end_date="2020-01-01",
            ),
            # Anachronism (FastAPI in 2012):
            Relation(
                subject_id="role:bad_role",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:fastapi",
                start_date="2012-01-01",
                end_date="2013-01-01",
            ),
        ],
    )

    report = validator.validate(payload)
    assert len(report.conflicts) >= 2
    assert report.penalty_factor >= 0.50
    assert report.is_valid is False

