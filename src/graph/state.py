"""
state.py
--------
Shared state that flows through the LangGraph pipeline. Every node reads
from and writes back into this single TypedDict.
"""

from typing import TypedDict, Optional

from src.extraction.extractor import ExtractedProfile
from src.scoring.scorer import ScoreBreakdown, ScoringWeights


class RankingState(TypedDict, total=False):
    # Inputs
    job_text: str
    cv_text: str
    weights: Optional[ScoringWeights]

    # Populated by nodes
    job_profile: Optional[ExtractedProfile]
    cv_profile: Optional[ExtractedProfile]
    score_breakdown: Optional[ScoreBreakdown]
    explanation: Optional[str]