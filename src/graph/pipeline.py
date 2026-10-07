"""
-----------
pipeline.py
-----------
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
)

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