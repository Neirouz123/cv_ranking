"""
state.py
--------
Shared state that flows through the LangGraph pipeline. Every node reads
from and writes back into this single TypedDict.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.extraction.extractor import ExtractedProfile
from src.scoring.scorer import ScoreBreakdown, ScoringWeights
from src.core.models import (
    CandidateRankingResult,
    ConflictReport,
    ExtractedGraph,
    GraphMatchBreakdown,
    JobDescriptionGraphPayload,
)


class RankingState(TypedDict, total=False):
    # Inputs
    job_text: str
    cv_text: str
    candidate_name: Optional[str]
    weights: Optional[ScoringWeights]

    # Populated by legacy nodes
    job_profile: Optional[ExtractedProfile]
    cv_profile: Optional[ExtractedProfile]
    score_breakdown: Optional[ScoreBreakdown]
    explanation: Optional[str]

    # Populated by Knowledge Graph nodes
    extracted_cv: Optional[ExtractedGraph]
    extracted_job: Optional[JobDescriptionGraphPayload]
    rdf_graph: Optional[Any]
    nx_graph: Optional[Any]
    conflict_report: Optional[ConflictReport]
    graph_breakdown: Optional[GraphMatchBreakdown]
    ranking_result: Optional[CandidateRankingResult]
    critique: Optional[str]