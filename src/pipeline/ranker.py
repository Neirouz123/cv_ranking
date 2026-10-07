"""
src/pipeline/ranker.py
----------------------
Hybrid multi-factor CV ranking engine.
Implements the objective ranking function:
    FinalScore = alpha * VectorSimilarity + beta * GraphMatchScore - gamma * ConflictPenalty
combining distributional textual vectors, topological knowledge graph metrics,
and consistency conflict penalties.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.models import (
    CandidateGraphPayload,
    CandidateRankingResult,
    JobDescriptionGraphPayload,
)
from src.core.validator import ConsistencyValidator
from src.graph.matching import TopologicalMatcher


class RankingWeights(BaseModel):
    """Configuration weights for the hybrid scoring function."""
    model_config = ConfigDict(frozen=False)

    alpha: float = Field(default=0.25, ge=0.0, description="Weight for textual Vector Similarity")
    beta: float = Field(default=0.60, ge=0.0, description="Weight for topological Graph Match Score")
    gamma: float = Field(default=0.20, ge=0.0, description="Penalty deduction factor for conflicts")


def compute_vector_similarity(text_a: Optional[str], text_b: Optional[str]) -> float:
    """
    Computes textual similarity between two documents combining
    TF-IDF cosine similarity and content token overlap.
    Returns a score in [0.0, 1.0].
    """
    if not text_a or not text_b:
        return 0.0

    clean_a = text_a.strip()
    clean_b = text_b.strip()

    if not clean_a or not clean_b:
        return 0.0

    tfidf_sim = 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([clean_a, clean_b])
        tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        pass

    # Content word overlap
    words_a = set(clean_a.lower().split())
    words_b = set(clean_b.lower().split())
    stopwords = {"and", "or", "in", "with", "the", "a", "an", "for", "to", "of", "on", "at", "by", "is", "as", "needed"}
    content_a = {w.strip(".,;:()") for w in words_a} - stopwords
    content_b = {w.strip(".,;:()") for w in words_b} - stopwords

    overlap_sim = 0.0
    if content_a and content_b:
        overlap_sim = len(content_a & content_b) / len(content_a | content_b)

    sim = max(tfidf_sim, overlap_sim)
    return min(1.0, max(0.0, sim))


class HybridRanker:
    """
    Hybrid scoring and ranking engine combining textual vector similarity,
    topological graph matching (PPR, SKOS distance, Jaccard), and contradiction penalties.
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        topological_matcher: Optional[TopologicalMatcher] = None,
        validator: Optional[ConsistencyValidator] = None,
    ) -> None:
        self.weights = weights or RankingWeights()
        self.matcher = topological_matcher or TopologicalMatcher()
        self.validator = validator or ConsistencyValidator()

    def score_candidate(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
    ) -> CandidateRankingResult:
        """
        Evaluate a single candidate against a job description.
        Applies:
            FinalScore = alpha * VectorSimilarity + beta * GraphMatchScore - gamma * ConflictPenalty
        """
        # 1. Textual Vector Similarity
        vec_sim = compute_vector_similarity(candidate_payload.raw_text, jd_payload.raw_text)

        # 2. Graph Match Score (PPR + SKOS Taxpath Distance + Jaccard)
        graph_breakdown = self.matcher.compute_match(candidate_payload, jd_payload)
        graph_score = graph_breakdown.graph_match_score

        # 3. Conflict Detection and Penalty
        val_report = self.validator.validate(candidate_payload)
        penalty = val_report.penalty_factor

        # 4. Multi-factor synthesis
        # Calculate raw normalized score in [0.0, 1.0]
        raw_composite = (
            self.weights.alpha * vec_sim
            + self.weights.beta * graph_score
            - self.weights.gamma * penalty
        )

        # Rescale to 0 - 100
        # If alpha + beta < 1, normalize by sum of positive weights
        pos_weight_sum = self.weights.alpha + self.weights.beta
        if pos_weight_sum > 0:
            normalized_base = (self.weights.alpha * vec_sim + self.weights.beta * graph_score) / pos_weight_sum
            # Apply penalty proportionally
            final_composite = max(0.0, normalized_base - (self.weights.gamma * penalty))
        else:
            final_composite = max(0.0, raw_composite)

        final_score_100 = round(min(100.0, max(0.0, final_composite * 100.0)), 2)

        return CandidateRankingResult(
            candidate_id=candidate_payload.candidate_id,
            candidate_name=candidate_payload.name,
            final_score=final_score_100,
            vector_similarity=round(vec_sim, 4),
            graph_match_score=round(graph_score, 4),
            conflict_penalty=round(penalty, 4),
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
        results = [self.score_candidate(cand, jd_payload) for cand in candidates]
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results
