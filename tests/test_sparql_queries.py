"""
tests/test_sparql_queries.py
----------------------------
Unit tests verifying SPARQL queries on in-memory RDFLib knowledge graphs:
1. `get_skills_for_role`: Traverse cv:hasExperience -> cv:usesSkill filtered by role pattern.
2. `get_low_confidence_relations`: Audit reified rdf:Statement entries with confidence < threshold.
3. `match_skills_with_taxonomy`: Transitive property path (skos:broader*) taxonomic roll-up.
"""

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, XSD

from src.core.ontology import CV, SKILL, build_skos_ontology
from src.graph.sparql_queries import (
    get_low_confidence_relations,
    get_skills_for_role,
    match_skills_with_taxonomy,
)


@pytest.fixture
def sample_rdf_graph() -> Graph:
    """Build an in-memory test graph with roles, skills, reified statements, and SKOS hierarchies."""
    g = build_skos_ontology()

    cand_uri = CV["candidate_1"]
    role_ml = CV["role_ml_engineer"]
    role_devops = CV["role_devops_specialist"]

    skill_pytorch = SKILL["pytorch"]
    skill_python = SKILL["python"]
    skill_docker = SKILL["docker"]

    # Candidate experiences
    g.add((cand_uri, CV.hasExperience, role_ml))
    g.add((role_ml, RDFS.label, Literal("Senior Machine Learning Engineer", datatype=XSD.string)))
    g.add((role_ml, CV.usesSkill, skill_pytorch))
    g.add((role_ml, CV.usesSkill, skill_python))

    g.add((cand_uri, CV.hasExperience, role_devops))
    g.add((role_devops, RDFS.label, Literal("DevOps Specialist", datatype=XSD.string)))
    g.add((role_devops, CV.usesSkill, skill_docker))

    # Add reified statements with different confidence levels
    stmt1 = CV["stmt_high_conf"]
    g.add((stmt1, RDF.type, RDF.Statement))
    g.add((stmt1, RDF.subject, role_ml))
    g.add((stmt1, RDF.predicate, CV.usesSkill))
    g.add((stmt1, RDF.object, skill_pytorch))
    g.add((stmt1, CV.confidence, Literal(0.95, datatype=XSD.float)))
    g.add((stmt1, CV.sourceSnippet, Literal("Lead engineer designing PyTorch pipelines.", datatype=XSD.string)))

    stmt2 = CV["stmt_low_conf_1"]
    g.add((stmt2, RDF.type, RDF.Statement))
    g.add((stmt2, RDF.subject, role_ml))
    g.add((stmt2, RDF.predicate, CV.usesSkill))
    g.add((stmt2, RDF.object, skill_python))
    g.add((stmt2, CV.confidence, Literal(0.55, datatype=XSD.float)))
    g.add((stmt2, CV.sourceSnippet, Literal("Briefly used scripting languages.", datatype=XSD.string)))

    stmt3 = CV["stmt_low_conf_2"]
    g.add((stmt3, RDF.type, RDF.Statement))
    g.add((stmt3, RDF.subject, role_devops))
    g.add((stmt3, RDF.predicate, CV.usesSkill))
    g.add((stmt3, RDF.object, skill_docker))
    g.add((stmt3, CV.confidence, Literal(0.65, datatype=XSD.float)))
    g.add((stmt3, CV.sourceSnippet, Literal("Docker mentioned in hobbies.", datatype=XSD.string)))

    return g


def test_get_skills_for_role(sample_rdf_graph):
    """Test SPARQL traversal finding all skills bound to roles matching a regex."""
    # Query Machine Learning role skills
    ml_skills = get_skills_for_role(sample_rdf_graph, role_regex="Machine Learning")
    assert len(ml_skills) >= 2

    skill_names = [s["skill_label"].lower() for s in ml_skills]
    assert any("pytorch" in s for s in skill_names)
    assert any("python" in s for s in skill_names)

    # Query DevOps role skills
    devops_skills = get_skills_for_role(sample_rdf_graph, role_regex="DevOps")
    assert len(devops_skills) >= 1
    devops_skill_names = [s["skill_label"].lower() for s in devops_skills]
    assert any("docker" in s for s in devops_skill_names)


def test_get_low_confidence_relations(sample_rdf_graph):
    """Test auditing reified rdf:Statement entries below confidence threshold."""
    low_conf = get_low_confidence_relations(sample_rdf_graph, threshold=0.70)
    assert len(low_conf) == 2

    confidences = [item["confidence"] for item in low_conf]
    assert all(c < 0.70 for c in confidences)
    assert 0.55 in confidences
    assert 0.65 in confidences

    # High confidence (0.95) must NOT be returned
    assert 0.95 not in confidences

    # Check sourceSnippet is retrieved
    snippets = [item["source_snippet"] for item in low_conf]
    assert any("scripting languages" in s for s in snippets)
    assert any("Docker mentioned" in s for s in snippets)


def test_match_skills_with_taxonomy(sample_rdf_graph):
    """Test SPARQL transitive property path (skos:broader*) roll-up to high-level parent."""
    # In ontology: PyTorch -> DeepLearning -> MachineLearning
    # PyTorch should match parent 'machinelearning'
    matches = match_skills_with_taxonomy(sample_rdf_graph, target_parent_skill="machinelearning")
    assert len(matches) >= 1

    matched_skills = [m["skill"].lower() for m in matches]
    assert any("pytorch" in s for s in matched_skills)

    # Test rolling up to deeplearning
    dl_matches = match_skills_with_taxonomy(sample_rdf_graph, target_parent_skill="deeplearning")
    assert len(dl_matches) >= 1
    dl_skills = [m["skill"].lower() for m in dl_matches]
    assert any("pytorch" in s for s in dl_skills)

