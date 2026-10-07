"""
-----------
pipeline.py
-----------
LangGraph pipelines orchestrating both:
1. Legacy text/regex extraction and scoring pipeline (graph).
2. Semantic Knowledge Graph, Validation & GraphRAG pipeline (kg_pipeline):
   extract_node -> graph_construction_node -> validation_node -> ranking_node.
"""

from langgraph.graph import StateGraph, START, END

from src.graph.state import RankingState
from src.graph.nodes import (
    extract_job,
    extract_cv,
    check_extraction_confidence,
    llm_extract_cv,
    compute_score,
    generate_explanation,
    extract_node,
    graph_construction_node,
    validation_node,
    ranking_node,
)

# --------------------------------------------------------------------------
# 1. Baseline Extraction & Rule-based Scoring Pipeline
# --------------------------------------------------------------------------
builder = StateGraph(RankingState)

builder.add_node("extract_job", extract_job)
builder.add_node("extract_cv", extract_cv)
builder.add_node("llm_extract_cv", llm_extract_cv)
builder.add_node("compute_score", compute_score)
builder.add_node("generate_explanation", generate_explanation)

builder.add_edge(START, "extract_job")
builder.add_edge(START, "extract_cv")

builder.add_conditional_edges(
    "extract_cv",
    check_extraction_confidence,
    {
        "llm_extract_cv": "llm_extract_cv",
        "compute_score": "compute_score",
    },
)
builder.add_edge("llm_extract_cv", "compute_score")

builder.add_edge("extract_job", "compute_score")
builder.add_edge("compute_score", "generate_explanation")
builder.add_edge("generate_explanation", END)

graph = builder.compile()


# --------------------------------------------------------------------------
# 2. Knowledge Graph, SPARQL & GraphRAG Stateful Orchestration Pipeline:
# extract_node -> graph_construction_node -> validation_node -> ranking_node
# --------------------------------------------------------------------------
kg_builder = StateGraph(RankingState)

kg_builder.add_node("extract_node", extract_node)
kg_builder.add_node("graph_construction_node", graph_construction_node)
kg_builder.add_node("validation_node", validation_node)
kg_builder.add_node("ranking_node", ranking_node)

kg_builder.add_edge(START, "extract_node")
kg_builder.add_edge("extract_node", "graph_construction_node")
kg_builder.add_edge("graph_construction_node", "validation_node")
kg_builder.add_edge("validation_node", "ranking_node")
kg_builder.add_edge("ranking_node", END)

kg_pipeline = kg_builder.compile()