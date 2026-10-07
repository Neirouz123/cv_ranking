"""
tests/test_ranking_pipeline.py
------------------------------
End-to-end synthetic pipeline test from raw triplet extraction to final scored rank:
- Extraction to ExtractedGraph
- RDFLib & NetworkX ingestion
- Pre-scoring conflict detection (ConflictReport)
- Topological scoring (PPR, weighted taxonomic distance)
- Hybrid rank computation: compute_final_rank(vector, graph, penalty, weights=(0.4, 0.5, 0.1))
- GraphRAG Louvain community prompt generation
"""

import pytest

from src.core.models import (
    Entity,
    EntityCategory,
    ExtractedGraph,
    JobDescriptionGraphPayload,
    PredicateType,
    Relation,
)
from src.core.ontology import calculate_taxonomic_distance, taxonomic_similarity
from src.core.validator import ConsistencyValidator
from src.graph.builder import GraphBuilder
from src.graph.graph_rag import GraphRAGSummaryEngine
from src.graph.matching import TopologicalMatcher
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker, compute_final_rank


def test_end_to_end_synthetic_ranking_pipeline():
    """
    Simulate full lifecycle:
    Raw Text -> ExtractedGraph -> RDF/NetworkX -> Validation -> PPR/Taxonomy -> Hybrid Score -> GraphRAG
    """
    raw_cv = """Geoffrey Hinton
Professional Experience
2019-01 to 2023-12: Principal Deep Learning Scientist at Cognitive AI
Spearheaded transformer and deep learning research with PyTorch and Python.
Architected scalable inference engines utilizing Docker and Kubernetes.

Education
2014-09 to 2018-06: PhD in Artificial Intelligence
"""

    raw_jd = """Senior Machine Learning Engineer
Requirements:
Deep expertise in Machine Learning and Deep Learning.
Hands-on proficiency in Python and Kubernetes.
"""

    # 1. Extraction step
    extractor = KnowledgeGraphExtractor(use_llm=False)
    extracted_cand = extractor.extract_candidate(raw_cv, candidate_id="cand_hinton", candidate_name="Geoffrey Hinton")
    assert isinstance(extracted_cand, ExtractedGraph)
    assert extracted_cand.candidate_id == "cand_hinton"
    assert len(extracted_cand.get_skills()) >= 3

    extracted_jd = extractor.extract_job(raw_jd, job_id="job_sr_ml", title="Senior Machine Learning Engineer")
    assert len(extracted_jd.get_skills()) >= 2

    # 2. Graph Ingestion step
    builder = GraphBuilder(include_ontology_taxonomies=True)
    rdf_g, nx_g = builder.build_candidate_graph(extracted_cand)
    assert rdf_g is not None
    assert nx_g.number_of_nodes() > 0
    assert nx_g.number_of_edges() > 0

    # 3. Conflict Detection step
    validator = ConsistencyValidator()
    conflict_report = validator.validate_conflicts(extracted_cand)
    assert conflict_report.penalty_score == 0.0
    assert conflict_report.is_valid is True
    assert len(conflict_report.conflicts) == 0

    # 4. Topological Match step
    matcher = TopologicalMatcher()
    breakdown = matcher.compute_match(extracted_cand, extracted_jd)
    assert breakdown.graph_match_score > 0.6
    assert breakdown.ppr_score > 0.0

    # Verify taxonomical inference (PyTorch matches Deep Learning)
    inferred_jd_skills = [m["jd_skill"] for m in breakdown.matched_skills_inferred]
    assert "Deep Learning" in inferred_jd_skills or "Machine Learning" in inferred_jd_skills

    # 5. Hybrid Multi-Factor Scoring with compute_final_rank
    vector_score = 0.50
    graph_score = breakdown.graph_match_score
    penalty_score = conflict_report.penalty_score

    final_rank_score = compute_final_rank(
        vector_score=vector_score,
        graph_score=graph_score,
        penalty_score=penalty_score,
        weights=(0.4, 0.5, 0.1),
    )
    assert 0.0 < final_rank_score <= 1.0
    # With 0 penalty, score should be >= 0.4*0.5 + 0.5*0.6 = 0.5
    assert final_rank_score >= 0.50

    # 6. GraphRAG Community Extraction & Prompt Generation
    rag_engine = GraphRAGSummaryEngine(use_llm=False)
    community_prompts = rag_engine.generate_community_prompts(extracted_cand)
    assert len(community_prompts) >= 1
    assert "Cluster" in community_prompts[0]["cluster_title"]
    assert "Prompt" in community_prompts[0]["prompt_input"] or "Cluster" in community_prompts[0]["prompt_input"]

    critique = rag_engine.generate_candidate_critique(extracted_cand, extracted_jd)
    assert "Candidate Fit Audit: Geoffrey Hinton" in critique
    assert "GraphRAG Knowledge Communities" in critique


def test_ranking_penalizes_contradictions_with_compute_final_rank():
    """Verify that compute_final_rank deducts penalties when contradictions are present."""
    vector_score = 0.80
    graph_score = 0.85

    # Case A: Clean candidate (0 penalty)
    score_clean = compute_final_rank(vector_score, graph_score, penalty_score=0.0, weights=(0.4, 0.5, 0.1))

    # Case B: Inconsistent candidate (penalty = 0.60)
    score_conflicted = compute_final_rank(vector_score, graph_score, penalty_score=0.60, weights=(0.4, 0.5, 0.1))

    assert score_clean > score_conflicted
    expected_diff = 0.1 * 0.60
    assert pytest.approx(score_clean - score_conflicted, abs=1e-4) == expected_diff

