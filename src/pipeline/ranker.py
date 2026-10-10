"""
src/pipeline/ranker.py
----------------------
Hybrid multi-factor CV ranking engine.
Implements the unified, strictly normalized objective ranking function:
    BaseScore = (alpha * S_vector + beta * S_graph + delta * S_exp + epsilon * S_degree) / (alpha + beta + delta + epsilon)
    FinalScore = max(0.0, min(1.0, BaseScore - gamma * PenaltyScore)) * 100
With:
- Capping at 60.0% if any mandatory (must-have) requirement is missing.
- Centralized calculation in `compute_final_rank`.
- Reusable bilingual `CorpusVectorizer`.
- Exact 100.0 score achievement for ideal profiles without conflicts.
"""

from __future__ import annotations

from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.core.models import (
    CandidateGraphPayload,
    CandidateRankingResult,
    JobDescriptionGraphPayload,
)
from src.core.validator import ConsistencyValidator
from src.graph.matching import TopologicalMatcher
from src.pipeline.vector_store import CorpusVectorizer


class RankingWeights(BaseModel):
    """Configuration weights for the hybrid scoring function."""
    model_config = ConfigDict(frozen=False)

    alpha: float = Field(default=0.25, ge=0.0, description="Weight for textual Vector Similarity (S_vector)")
    beta: float = Field(default=0.45, ge=0.0, description="Weight for topological Graph Match Score (S_graph)")
    delta: float = Field(default=0.15, ge=0.0, description="Weight for Experience / Seniority (S_exp)")
    epsilon: float = Field(default=0.15, ge=0.0, description="Weight for Academic Degree qualification (S_degree)")
    gamma: float = Field(default=0.25, ge=0.0, description="Penalty deduction factor for conflicts (PenaltyScore)")


def compute_vector_similarity(text_a: Optional[str], text_b: Optional[str]) -> float:
    """
    Computes textual similarity between two documents using CorpusVectorizer.
    Returns a score in [0.0, 1.0].
    """
    vectorizer = CorpusVectorizer()
    return vectorizer.compute_similarity(text_a, text_b)


def compute_final_rank(
    vector_score: float,
    graph_score: float,
    penalty_score: float = 0.0,
    exp_score: float = 1.0,
    degree_score: float = 1.0,
    weights: Optional[Union[RankingWeights, tuple]] = None,
    has_missing_must: bool = False,
    scale_100: Optional[bool] = None,
) -> float:
    """
    Unified, single source of truth for composite candidate scoring:
        BaseScore = (alpha * S_vector + beta * S_graph + delta * S_exp + epsilon * S_degree) / (alpha + beta + delta + epsilon)
        FinalScore = max(0.0, min(1.0, BaseScore - gamma * PenaltyScore)) * 100

    If has_missing_must is True, final score is capped at 60.0% (or 0.60).
    Handles edge case alpha + beta + delta + epsilon == 0 safely by falling back to 0.0.
    """
    is_legacy_tuple = isinstance(weights, tuple) and len(weights) == 3
    should_scale = (scale_100 if scale_100 is not None else not is_legacy_tuple)

    if weights is None:
        rw = RankingWeights()
        alpha, beta, delta, epsilon, gamma = rw.alpha, rw.beta, rw.delta, rw.epsilon, rw.gamma
    elif isinstance(weights, RankingWeights):
        alpha, beta, delta, epsilon, gamma = weights.alpha, weights.beta, weights.delta, weights.epsilon, weights.gamma
    elif isinstance(weights, tuple):
        if len(weights) == 3:
            # Legacy 3-tuple: (w_vector, w_graph, w_penalty)
            alpha, beta, gamma = float(weights[0]), float(weights[1]), float(weights[2])
            delta, epsilon = 0.0, 0.0
            if not should_scale:
                # Direct unscaled dot-product for backwards-compatibility with unscaled tests
                raw_legacy = alpha * vector_score + beta * graph_score - gamma * penalty_score
                return float(min(1.0, max(0.0, raw_legacy)))
        elif len(weights) == 5:
            alpha, beta, delta, epsilon, gamma = (
                float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3]), float(weights[4])
            )
        else:
            raise ValueError(f"Unsupported weights tuple length: {len(weights)}")
    else:
        rw = RankingWeights()
        alpha, beta, delta, epsilon, gamma = rw.alpha, rw.beta, rw.delta, rw.epsilon, rw.gamma

    # Base score with strict normalization
    pos_sum = alpha + beta + delta + epsilon
    if pos_sum > 0.0:
        base_score = (
            alpha * vector_score
            + beta * graph_score
            + delta * exp_score
            + epsilon * degree_score
        ) / pos_sum
    else:
        base_score = 0.0

    # Apply conflict deduction penalty
    composite_0_1 = max(0.0, min(1.0, base_score - (gamma * penalty_score)))

    # Hard gate capping for missing mandatory requirements
    if has_missing_must:
        composite_0_1 = min(composite_0_1, 0.60)

    if should_scale:
        return round(composite_0_1 * 100.0, 2)
    return float(composite_0_1)


class HybridRanker:
    """
    Hybrid scoring and ranking engine combining textual vector similarity,
    topological graph matching (PPR, SKOS distance, Jaccard), seniority,
    degree qualification, and contradiction penalties.
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        topological_matcher: Optional[TopologicalMatcher] = None,
        validator: Optional[ConsistencyValidator] = None,
        vectorizer: Optional[CorpusVectorizer] = None,
    ) -> None:
        self.weights = weights or RankingWeights()
        self.matcher = topological_matcher or TopologicalMatcher()
        self.validator = validator or ConsistencyValidator()
        self.vectorizer = vectorizer or CorpusVectorizer()

    def score_candidate(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
    ) -> CandidateRankingResult:
        """
        Evaluate a single candidate against a job description using the unified
        `compute_final_rank` formula.
        """
        # 1. Textual Vector Similarity (CorpusVectorizer)
        vec_sim = self.vectorizer.compute_similarity(candidate_payload.raw_text, jd_payload.raw_text)

        # 2. Graph Match Score (PPR + SKOS Taxpath Distance + Jaccard)
        graph_breakdown = self.matcher.compute_match(candidate_payload, jd_payload)
        graph_score = graph_breakdown.graph_match_score

        # 3. Experience & Seniority Score
        cand_exp = candidate_payload.compute_active_career_duration_years()
        req_exp = jd_payload.required_experience_years
        if req_exp is None or req_exp <= 0.0:
            exp_score = 1.0
        else:
            exp_score = min(1.0, cand_exp / float(req_exp))

        # 4. Academic Degree Score
        cand_deg = candidate_payload.resolve_degree_level()
        req_deg = jd_payload.required_degree_level
        if req_deg is None or req_deg <= 0:
            deg_score = 1.0
        else:
            if cand_deg >= req_deg:
                deg_score = 1.0
            else:
                deg_score = round(cand_deg / float(req_deg), 2)

        # 5. Conflict Detection and Penalty
        val_report = self.validator.validate(candidate_payload)
        penalty = val_report.penalty_factor

        # 6. Must-have / Hard Gate check
        missing_must = graph_breakdown.missing_must_skills
        has_missing = len(missing_must) > 0

        # 7. Unified composite scoring via single source of truth
        final_score_100 = compute_final_rank(
            vector_score=vec_sim,
            graph_score=graph_score,
            penalty_score=penalty,
            exp_score=exp_score,
            degree_score=deg_score,
            weights=self.weights,
            has_missing_must=has_missing,
            scale_100=True,
        )

        is_capped = bool(has_missing and final_score_100 <= 60.0)

        return CandidateRankingResult(
            candidate_id=candidate_payload.candidate_id,
            candidate_name=candidate_payload.name,
            final_score=final_score_100,
            vector_similarity=round(vec_sim, 4),
            graph_match_score=round(graph_score, 4),
            experience_score=round(exp_score, 4),
            degree_score=round(deg_score, 4),
            conflict_penalty=round(penalty, 4),
            missing_must_requirements=missing_must,
            is_capped_by_must_have=is_capped,
            graph_breakdown=graph_breakdown,
            validation_report=val_report,
        )

    def rank_candidates(
        self,
        candidates: list[CandidateGraphPayload],
        jd_payload: JobDescriptionGraphPayload,
    ) -> list[CandidateRankingResult]:
        """
        Score and rank a collection of candidates for a job description.
        Returns candidate records sorted by final_score descending.
        """
        # Pre-fit vectorizer across all candidate texts + JD simultaneously
        if candidates and jd_payload.raw_text:
            cand_texts = [c.raw_text or "" for c in candidates]
            self.vectorizer.compute_batch_similarities(jd_payload.raw_text, cand_texts)

        results = [self.score_candidate(cand, jd_payload) for cand in candidates]
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results
