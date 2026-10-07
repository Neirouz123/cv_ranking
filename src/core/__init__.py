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
    ConflictReport,
    ExtractedGraph,
    GraphMatchBreakdown,
    CandidateRankingResult,
)
from src.core.validator import validate_candidate_graph

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
    "ConflictReport",
    "ExtractedGraph",
    "GraphMatchBreakdown",
    "CandidateRankingResult",
    "validate_candidate_graph",
]

