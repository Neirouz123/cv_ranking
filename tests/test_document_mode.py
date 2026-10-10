"""
tests/test_document_mode.py
---------------------------
Unit tests for:
1. Lazy-loading ChatGroq & fallback provider (get_llm_client).
2. Technical Document extraction (extract_document) on RFC 7540 and MQTT 5.0.
3. Document RDF & NetworkX graph building.
4. Specification contradiction and consistency validation (validate_document_graph).
5. Pre-configured document SPARQL queries.
"""

from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest
from rdflib import Graph

from src.core.models import (
    ConflictType,
    Entity,
    EntityCategory,
    ExtractedGraph,
    PredicateType,
    Relation,
)
from src.core.validator import validate_document_graph
from src.graph.builder import GraphBuilder
from src.graph.sparql_queries import (
    find_circular_constraints,
    get_defined_parameters,
    get_document_dependencies,
    get_protocol_conflicts,
)
from src.pipeline.extractor import KnowledgeGraphExtractor, get_llm_client

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_documents"


def test_get_llm_client_returns_none_when_no_key():
    with patch.dict("os.environ", {}, clear=True):
        client = get_llm_client(api_key=None)
        assert client is None


def test_get_llm_client_lazy_loads_with_key():
    client = get_llm_client(api_key="gsk_test_mock_key_12345")
    assert client is not None
    assert hasattr(client, "invoke")


def test_offline_extractor_initializes_without_groq_key():
    with patch.dict("os.environ", {}, clear=True):
        extractor = KnowledgeGraphExtractor(use_llm=False)
        assert extractor._llm is None


def test_rfc7540_sample_extraction():
    sample_file = SAMPLE_DIR / "rfc7540_http2_framing.txt"
    assert sample_file.exists(), f"Missing sample file {sample_file}"

    text = sample_file.read_text(encoding="utf-8")
    extractor = KnowledgeGraphExtractor(use_llm=False)
    doc_graph = extractor.extract_document(text, document_id="rfc7540", document_title="RFC 7540 HTTP/2")

    assert len(doc_graph.entities) > 5
    assert len(doc_graph.relations) > 3

    # Check that sections, parameters, concepts, and specifications are present
    categories = {e.category for e in doc_graph.entities}
    assert EntityCategory.SECTION in categories
    assert EntityCategory.PARAMETER in categories
    assert EntityCategory.SPECIFICATION in categories or EntityCategory.PROTOCOL in categories

    # Check relation predicates
    predicates = {r.predicate for r in doc_graph.relations}
    assert PredicateType.DEFINES in predicates
    assert PredicateType.DEPENDS_ON in predicates
    assert PredicateType.CONFLICTS_WITH in predicates


def test_mqtt5_sample_extraction():
    sample_file = SAMPLE_DIR / "mqtt_qos_specification.txt"
    assert sample_file.exists()

    text = sample_file.read_text(encoding="utf-8")
    extractor = KnowledgeGraphExtractor(use_llm=False)
    doc_graph = extractor.extract_document(text, document_id="mqtt5", document_title="MQTT 5.0")

    predicates = {r.predicate for r in doc_graph.relations}
    assert PredicateType.DEFINES in predicates
    assert PredicateType.DEPENDS_ON in predicates
    assert PredicateType.CONFLICTS_WITH in predicates


def test_document_graph_builder_ingestion():
    sample_file = SAMPLE_DIR / "rfc7540_http2_framing.txt"
    text = sample_file.read_text(encoding="utf-8")
    extractor = KnowledgeGraphExtractor(use_llm=False)
    doc_graph = extractor.extract_document(text, document_id="rfc7540")

    builder = GraphBuilder(include_ontology_taxonomies=False)
    rdf_g, nx_g = builder.build_candidate_graph(doc_graph)

    assert isinstance(rdf_g, Graph)
    assert isinstance(nx_g, nx.MultiDiGraph)
    assert len(rdf_g) > 10
    assert nx_g.number_of_nodes() > 5
    assert nx_g.number_of_edges() > 3


def test_validate_document_graph_detects_conflicts_and_mismatches():
    sample_file = SAMPLE_DIR / "rfc7540_http2_framing.txt"
    text = sample_file.read_text(encoding="utf-8")
    extractor = KnowledgeGraphExtractor(use_llm=False)
    doc_graph = extractor.extract_document(text, document_id="rfc7540")

    report = validate_document_graph(doc_graph)
    assert not report.is_valid
    assert len(report.conflicts) > 0

    c_types = {c["conflict_type"] for c in report.conflicts}
    # Should detect both protocol conflicts and conflicting window size definitions across sections
    assert ConflictType.SPECIFICATION_CONFLICT.value in c_types
    assert ConflictType.PARAMETER_MISMATCH.value in c_types


def test_validate_document_graph_detects_circular_dependencies():
    cycle_graph = ExtractedGraph(
        candidate_id="cycle_doc",
        name="Circular Standard",
        entities=[
            Entity(id="concept:a", name="Protocol A", category=EntityCategory.CONCEPT),
            Entity(id="concept:b", name="Protocol B", category=EntityCategory.CONCEPT),
        ],
        relations=[
            Relation(subject_id="concept:a", predicate=PredicateType.DEPENDS_ON, object_id="concept:b"),
            Relation(subject_id="concept:b", predicate=PredicateType.DEPENDS_ON, object_id="concept:a"),
        ],
    )

    report = validate_document_graph(cycle_graph)
    assert not report.is_valid
    c_types = {c["conflict_type"] for c in report.conflicts}
    assert ConflictType.CIRCULAR_DEPENDENCY.value in c_types


def test_validate_document_graph_clean_specification():
    clean_graph = ExtractedGraph(
        candidate_id="clean_doc",
        name="Clean Specification",
        entities=[
            Entity(id="sec_1", name="Section 1", category=EntityCategory.SECTION),
            Entity(id="param_timeout", name="Timeout", category=EntityCategory.PARAMETER, metadata={"value": "30s"}),
            Entity(id="concept:http", name="HTTP", category=EntityCategory.CONCEPT),
            Entity(id="spec:tcp", name="TCP", category=EntityCategory.SPECIFICATION),
        ],
        relations=[
            Relation(subject_id="sec_1", predicate=PredicateType.DEFINES, object_id="param_timeout", metadata={"value": "30s"}),
            Relation(subject_id="concept:http", predicate=PredicateType.DEPENDS_ON, object_id="spec:tcp"),
        ],
    )

    report = validate_document_graph(clean_graph)
    assert report.is_valid
    assert len(report.conflicts) == 0
    assert report.penalty_score == 0.0


def test_document_sparql_queries():
    sample_file = SAMPLE_DIR / "rfc7540_http2_framing.txt"
    text = sample_file.read_text(encoding="utf-8")
    extractor = KnowledgeGraphExtractor(use_llm=False)
    doc_graph = extractor.extract_document(text, document_id="rfc7540")

    builder = GraphBuilder(include_ontology_taxonomies=False)
    rdf_g, _ = builder.build_candidate_graph(doc_graph)

    # 1. Dependencies query
    deps = get_document_dependencies(rdf_g)
    assert len(deps) > 0
    assert any("concept" in d and "specification" in d for d in deps)

    # 2. Protocol conflicts query
    confs = get_protocol_conflicts(rdf_g)
    assert len(confs) > 0
    assert any("protocol_a" in c and "protocol_b" in c for c in confs)

    # 3. Defined parameters query
    params = get_defined_parameters(rdf_g)
    assert len(params) > 0
    assert any("section" in p and "parameter" in p for p in params)

    # 4. Circular constraints query (should be empty for RFC 7540)
    circs = find_circular_constraints(rdf_g)
    assert isinstance(circs, list)
