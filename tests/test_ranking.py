"""
tests/test_ranking.py
---------------------
Unit tests for topological scoring, SKOS taxpath distance,
Personalized PageRank, and hybrid multi-factor ranking.
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
from src.core.ontology import calculate_taxonomic_distance, taxonomic_similarity
from src.graph.graph_rag import GraphRAGSummaryEngine
from src.graph.matching import TopologicalMatcher
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker, RankingWeights


@pytest.fixture
def target_jd() -> JobDescriptionGraphPayload:
    """JD requiring Deep Learning, Python, and Cloud/DevOps."""
    return JobDescriptionGraphPayload(
        job_id="job_lead_ai",
        title="Lead Deep Learning Engineer",
        entities=[
            Entity(id="job_lead_ai", label="Lead Deep Learning Engineer", category=EntityCategory.ROLE),
            Entity(id="skill:deeplearning", label="Deep Learning", category=EntityCategory.SKILL),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:kubernetes", label="Kubernetes", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(subject_id="job_lead_ai", predicate=PredicateType.USED_SKILL, object_id="skill:deeplearning"),
            Relation(subject_id="job_lead_ai", predicate=PredicateType.USED_SKILL, object_id="skill:python"),
            Relation(subject_id="job_lead_ai", predicate=PredicateType.USED_SKILL, object_id="skill:kubernetes"),
        ],
        raw_text="Lead Deep Learning Engineer needed. Must have expertise in Deep Learning, Python, and Kubernetes.",
    )


def test_ontological_distance_and_partial_credit():
    """Verify SKOS shortest path distance awards partial credit for related skills."""
    # PyTorch -> DeepLearning is 1 hop (narrower/broader)
    dist_pytorch_dl = calculate_taxonomic_distance("skill:pytorch", "skill:deeplearning")
    assert dist_pytorch_dl == 1.0

    sim_pytorch_dl = taxonomic_similarity("skill:pytorch", "skill:deeplearning")
    assert sim_pytorch_dl > 0.6  # Partial credit awarded

    # PyTorch -> TensorFlow is 2 hops (via DeepLearning)
    dist_pytorch_tf = calculate_taxonomic_distance("skill:pytorch", "skill:tensorflow")
    assert dist_pytorch_tf == 2.0
    sim_pytorch_tf = taxonomic_similarity("skill:pytorch", "skill:tensorflow")
    assert 0.0 < sim_pytorch_tf < sim_pytorch_dl  # Lower partial credit for 2 hops

    # Unrelated skills (e.g. PyTorch and Frontend React)
    dist_unrelated = calculate_taxonomic_distance("skill:pytorch", "skill:react")
    assert dist_unrelated > 3.0 or dist_unrelated == float("inf")


def test_clean_candidate_ranking_flow(target_jd):
    """A clean, highly qualified candidate should achieve a high final score with no penalties."""
    cand = CandidateGraphPayload(
        candidate_id="cand_strong",
        name="Dr. Alan Turing",
        entities=[
            Entity(id="cand_strong", label="Dr. Alan Turing", category=EntityCategory.CANDIDATE),
            Entity(id="role:ai_researcher", label="AI Researcher", category=EntityCategory.ROLE),
            # Has PyTorch (related to DeepLearning) and Python and Kubernetes
            Entity(id="skill:pytorch", label="PyTorch", category=EntityCategory.SKILL),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:kubernetes", label="Kubernetes", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(
                subject_id="cand_strong",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:ai_researcher",
                start_date="2020-01-01",
                end_date="2024-01-01",
            ),
            Relation(
                subject_id="role:ai_researcher",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:pytorch",
                start_date="2020-01-01",
                end_date="2024-01-01",
            ),
            Relation(
                subject_id="role:ai_researcher",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:python",
                start_date="2020-01-01",
                end_date="2024-01-01",
            ),
            Relation(
                subject_id="role:ai_researcher",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:kubernetes",
                start_date="2020-01-01",
                end_date="2024-01-01",
            ),
        ],
        raw_text="Dr. Alan Turing. AI Researcher with extensive experience in PyTorch, Python, and Kubernetes.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand, target_jd)

    assert result.final_score >= 60.0
    assert result.conflict_penalty == 0.0
    assert result.graph_breakdown.ontological_distance_score > 0.7
    assert "Python" in result.graph_breakdown.matched_skills_exact
    # PyTorch matched taxonomically to Deep Learning
    inferred_names = [m["jd_skill"] for m in result.graph_breakdown.matched_skills_inferred]
    assert "Deep Learning" in inferred_names


def test_contradictory_candidate_receives_penalty(target_jd):
    """A candidate with temporal contradictions must trigger penalties and lower the final score."""
    cand_contradictory = CandidateGraphPayload(
        candidate_id="cand_inconsistent",
        name="Chaotic Candidate",
        entities=[
            Entity(id="cand_inconsistent", label="Chaotic Candidate", category=EntityCategory.CANDIDATE),
            Entity(id="role:dev", label="Developer", category=EntityCategory.ROLE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL, claimed_years=10.0),
            Entity(id="skill:kubernetes", label="Kubernetes", category=EntityCategory.SKILL),
        ],
        relations=[
            # Temporal Inversion
            Relation(
                subject_id="cand_inconsistent",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:dev",
                start_date="2024-01-01",
                end_date="2021-01-01",
            ),
            # Over-claiming: claims 10 years, role is inverted/invalid
            Relation(
                subject_id="role:dev",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:python",
            ),
        ],
        raw_text="Chaotic Candidate, Developer with Python and Kubernetes.",
    )

    ranker = HybridRanker()
    result = ranker.score_candidate(cand_contradictory, target_jd)

    assert result.conflict_penalty > 0.0
    assert len(result.validation_report.conflicts) > 0
    # Penalty reduces the final score
    assert result.final_score < 50.0


def test_ppr_diffusion_and_community_critique(target_jd):
    """Verify PPR computation and GraphRAG Louvain community critique generation."""
    cand = CandidateGraphPayload(
        candidate_id="cand_graphrag",
        name="Grace Hopper",
        entities=[
            Entity(id="cand_graphrag", label="Grace Hopper", category=EntityCategory.CANDIDATE),
            Entity(id="role:chief_eng", label="Chief Engineer", category=EntityCategory.ROLE),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:fastapi", label="FastAPI", category=EntityCategory.SKILL),
            Entity(id="skill:docker", label="Docker", category=EntityCategory.SKILL),
            Entity(id="skill:kubernetes", label="Kubernetes", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(
                subject_id="cand_graphrag",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:chief_eng",
                start_date="2019-01-01",
                end_date="2023-12-31",
            ),
            Relation(
                subject_id="role:chief_eng",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:python",
                start_date="2019-01-01",
                end_date="2023-12-31",
            ),
            Relation(
                subject_id="role:chief_eng",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:fastapi",
                start_date="2019-01-01",
                end_date="2023-12-31",
            ),
            Relation(
                subject_id="role:chief_eng",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:docker",
                start_date="2019-01-01",
                end_date="2023-12-31",
            ),
            Relation(
                subject_id="role:chief_eng",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:kubernetes",
                start_date="2019-01-01",
                end_date="2023-12-31",
            ),
        ],
        raw_text="Grace Hopper. Chief Engineer with Python, FastAPI, Docker, and Kubernetes.",
    )

    # Test PPR via TopologicalMatcher
    matcher = TopologicalMatcher()
    breakdown = matcher.compute_match(cand, target_jd)
    assert breakdown.ppr_score > 0.0

    # Test GraphRAG summary engine
    engine = GraphRAGSummaryEngine(use_llm=False)
    clusters = engine.detect_communities(cand)
    assert len(clusters) >= 1

    critique = engine.generate_candidate_critique(cand, target_jd)
    assert "Candidate Fit Audit: Grace Hopper" in critique
    assert "GraphRAG Knowledge Communities" in critique
    assert "Career Trajectory & Continuity" in critique


def test_extractor_end_to_end_parsing(target_jd):
    """Test extractor creating a valid candidate knowledge graph from raw CV text."""
    extractor = KnowledgeGraphExtractor(use_llm=False)
    cv_text = """Marie Curie
Experience
2020 - 2023: Senior Data Scientist
Developed machine learning pipelines using PyTorch and Python.
Deployed models with Docker on Kubernetes.

Education
2015 - 2019: Master in Computer Science
"""
    payload = extractor.extract_candidate(cv_text, candidate_id="cand_curie", candidate_name="Marie Curie")
    assert payload.candidate_id == "cand_curie"
    assert payload.name == "Marie Curie"
    assert len(payload.get_skills()) >= 3

    # Ensure ranker can process extracted payload directly
    ranker = HybridRanker()
    result = ranker.score_candidate(payload, target_jd)
    assert result.final_score > 50.0
    assert result.conflict_penalty == 0.0


def test_multiple_candidate_ranking_order(target_jd):
    """Ensure consistent high-fit candidate ranks above contradictory or low-fit candidate."""
    cand_good = CandidateGraphPayload(
        candidate_id="cand_good",
        name="Good Candidate",
        entities=[
            Entity(id="cand_good", label="Good Candidate", category=EntityCategory.CANDIDATE),
            Entity(id="role:ml", label="ML Engineer", category=EntityCategory.ROLE),
            Entity(id="skill:deeplearning", label="Deep Learning", category=EntityCategory.SKILL),
            Entity(id="skill:python", label="Python", category=EntityCategory.SKILL),
            Entity(id="skill:kubernetes", label="Kubernetes", category=EntityCategory.SKILL),
        ],
        relations=[
            Relation(subject_id="cand_good", predicate=PredicateType.HELD_ROLE, object_id="role:ml", start_date="2020", end_date="2024"),
            Relation(subject_id="role:ml", predicate=PredicateType.USED_SKILL, object_id="skill:deeplearning", start_date="2020", end_date="2024"),
            Relation(subject_id="role:ml", predicate=PredicateType.USED_SKILL, object_id="skill:python", start_date="2020", end_date="2024"),
            Relation(subject_id="role:ml", predicate=PredicateType.USED_SKILL, object_id="skill:kubernetes", start_date="2020", end_date="2024"),
        ],
        raw_text="Good Candidate. Deep Learning, Python, and Kubernetes expert.",
    )

    cand_bad = CandidateGraphPayload(
        candidate_id="cand_bad",
        name="Inconsistent Candidate",
        entities=[
            Entity(id="cand_bad", label="Inconsistent Candidate", category=EntityCategory.CANDIDATE),
            Entity(id="role:junior", label="Junior", category=EntityCategory.ROLE),
            Entity(id="skill:fastapi", label="FastAPI", category=EntityCategory.SKILL),
        ],
        relations=[
            # Temporal Inversion & Anachronism
            Relation(subject_id="cand_bad", predicate=PredicateType.HELD_ROLE, object_id="role:junior", start_date="2024", end_date="2020"),
            Relation(subject_id="role:junior", predicate=PredicateType.USED_SKILL, object_id="skill:fastapi", start_date="2012", end_date="2013"),
        ],
        raw_text="Inconsistent Candidate with unrelated profile.",
    )

    ranker = HybridRanker()
    ranked = ranker.rank_candidates([cand_bad, cand_good], target_jd)

    assert len(ranked) == 2
    assert ranked[0].candidate_id == "cand_good"
    assert ranked[1].candidate_id == "cand_bad"
    assert ranked[0].final_score > ranked[1].final_score

