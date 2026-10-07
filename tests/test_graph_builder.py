"""
tests/test_graph_builder.py
---------------------------
Unit tests for knowledge graph ingestion and bi-directional conversion
between RDFLib semantic graphs and NetworkX MultiDiGraph structures.
"""

import pytest
import networkx as nx
from rdflib import Graph
from rdflib.namespace import RDF, SKOS

from src.core.models import (
    CandidateGraphPayload,
    Entity,
    EntityCategory,
    PredicateType,
    Relation,
)
from src.core.ontology import CV
from src.graph.builder import GraphBuilder


@pytest.fixture
def sample_payload() -> CandidateGraphPayload:
    return CandidateGraphPayload(
        candidate_id="cand_test",
        name="Ada Lovelace",
        entities=[
            Entity(id="cand_test", label="Ada Lovelace", category=EntityCategory.CANDIDATE),
            Entity(id="role:senior_architect", label="Senior AI Architect", category=EntityCategory.ROLE),
            Entity(id="company:babbage_labs", label="Babbage Labs", category=EntityCategory.COMPANY),
            Entity(id="skill:pytorch", label="PyTorch", category=EntityCategory.SKILL, claimed_years=4.5),
            Entity(id="degree:phd_cs", label="PhD Computer Science", category=EntityCategory.DEGREE),
        ],
        relations=[
            Relation(
                subject_id="cand_test",
                predicate=PredicateType.HELD_ROLE,
                object_id="role:senior_architect",
                start_date="2019-01-01",
                end_date="2023-12-31",
                confidence=0.98,
            ),
            Relation(
                subject_id="role:senior_architect",
                predicate=PredicateType.WORKED_AT,
                object_id="company:babbage_labs",
                start_date="2019-01-01",
                end_date="2023-12-31",
                confidence=0.95,
            ),
            Relation(
                subject_id="role:senior_architect",
                predicate=PredicateType.USED_SKILL,
                object_id="skill:pytorch",
                start_date="2020-03-01",
                end_date="2023-12-31",
                confidence=0.92,
            ),
            Relation(
                subject_id="cand_test",
                predicate=PredicateType.EARNED_DEGREE,
                object_id="degree:phd_cs",
                end_date="2018-06-30",
                confidence=1.0,
            ),
        ],
    )


def test_payload_to_rdflib_ingestion(sample_payload):
    """Verify entities, relations, and qualifiers are ingested into RDFLib."""
    builder = GraphBuilder(include_ontology_taxonomies=True)
    g = builder.payload_to_rdflib(sample_payload)

    assert isinstance(g, Graph)
    # Check that candidate entity exists
    cand_uri = CV["cand_test"]
    assert (cand_uri, RDF.type, CV.Candidate) in g

    # Check that statements / relations exist
    triples = list(g.triples((cand_uri, CV.heldRole, None)))
    assert len(triples) == 1

    # Check SKOS concepts were integrated
    skos_triples = list(g.triples((None, SKOS.broader, None)))
    assert len(skos_triples) > 0


def test_rdflib_to_networkx_preserves_attributes(sample_payload):
    """Verify RDFLib graph correctly exports to nx.MultiDiGraph preserving edge qualifiers."""
    builder = GraphBuilder(include_ontology_taxonomies=False)
    rdf_g = builder.payload_to_rdflib(sample_payload)
    nx_g = builder.rdflib_to_networkx(rdf_g)

    assert isinstance(nx_g, nx.MultiDiGraph)
    assert nx_g.number_of_nodes() >= 5
    assert nx_g.number_of_edges() >= 4

    # Check node categories
    assert nx_g.nodes["cand_test"]["category"] == "CANDIDATE"
    assert nx_g.nodes["skill:pytorch"]["category"] == "SKILL"
    assert nx_g.nodes["skill:pytorch"]["claimed_years"] == 4.5

    # Check edge attributes on usedSkill edge
    edge_found = False
    for u, v, data in nx_g.edges(data=True):
        if "senior_architect" in u and "pytorch" in v:
            edge_found = True
            assert data["predicate_type"] in {"USES_SKILL", "USED_SKILL"}
            assert data["start_date"] == "2020-03-01"
            assert data["end_date"] == "2023-12-31"
            assert data["confidence"] == 0.92
            break

    assert edge_found is True


def test_bidirectional_conversion_networkx_to_rdflib(sample_payload):
    """Verify converting nx.MultiDiGraph back into RDFLib maintains semantic structure."""
    builder = GraphBuilder(include_ontology_taxonomies=False)
    _, nx_g = builder.build_candidate_graph(sample_payload)

    # Convert back to RDFLib
    reconstructed_rdf = builder.networkx_to_rdflib(nx_g)
    assert isinstance(reconstructed_rdf, Graph)

    # Re-export to NetworkX to verify round-trip stability
    roundtrip_nx = builder.rdflib_to_networkx(reconstructed_rdf)

    assert roundtrip_nx.number_of_nodes() == nx_g.number_of_nodes()
    assert roundtrip_nx.number_of_edges() == nx_g.number_of_edges()

    # Verify attributes survived the round-trip
    cand_node = roundtrip_nx.nodes["cand_test"]
    assert cand_node["category"] == "CANDIDATE"
