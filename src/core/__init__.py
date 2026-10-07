"""
Core models and ontologies package for cv-ranking.
"""

from src.core.models import (
    Entity,
    EntityCategory,
    Relation,
    PredicateType,
    CandidateGraphPayload,
    JobDescriptionGraphPayload,
    Conflict,
    ConflictType,
    ConflictSeverity,
    ValidationReport,
    GraphMatchBreakdown,
    CandidateRankingResult,
)

__all__ = [
    "Entity",
    "EntityCategory",
    "Relation",
    "PredicateType",
    "CandidateGraphPayload",
    "JobDescriptionGraphPayload",
    "Conflict",
    "ConflictType",
    "ConflictSeverity",
    "ValidationReport",
    "GraphMatchBreakdown",
    "CandidateRankingResult",
]

