"""
src/graph/sparql_queries.py
---------------------------
Parameterized SPARQL query suite for auditing candidate RDF knowledge graphs.
Implements:
1. `get_skills_for_role`: Traverse cv:hasExperience -> cv:usesSkill filtered by role pattern.
2. `get_low_confidence_relations`: Audit reified rdf:Statement entries with confidence below threshold.
3. `match_skills_with_taxonomy`: Transitive property path (skos:broader*) taxonomic roll-up.
"""

from __future__ import annotations

import re
from typing import Any
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from src.core.ontology import CV, SKILL


def get_skills_for_role(graph: Graph, role_regex: str) -> list[dict[str, Any]]:
    """
    Traverse cv:hasExperience -> cv:usesSkill (and cv:heldRole) filtered by role title regex pattern.
    Returns:
        list of dicts with keys: 'role', 'role_label', 'skill', 'skill_label'
    """
    # Sanitize role regex for SPARQL
    safe_pattern = role_regex.replace("\\", "\\\\").replace('"', '\\"')

    sparql_query = f"""
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?role ?roleLabel ?skill ?skillLabel WHERE {{
        {{
            ?candidate (cv:hasExperience | cv:heldRole) ?role .
            ?role (cv:usesSkill | cv:usedSkill) ?skill .
        }} UNION {{
            ?role (cv:usesSkill | cv:usedSkill) ?skill .
        }}
        OPTIONAL {{ ?role rdfs:label ?roleLabel }}
        OPTIONAL {{ ?skill rdfs:label ?skillLabel }}
        FILTER(
            REGEX(STR(?roleLabel), "{safe_pattern}", "i") ||
            REGEX(STR(?role), "{safe_pattern}", "i")
        )
    }}
    """

    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []

    for row in results:
        role_uri = str(row.role) if row.role else ""
        skill_uri = str(row.skill) if row.skill else ""
        role_lbl = str(row.roleLabel) if row.roleLabel else role_uri.split("#")[-1].replace("_", " ")
        skill_lbl = str(row.skillLabel) if row.skillLabel else skill_uri.split("#")[-1].replace("_", " ")

        output.append({
            "role": role_uri,
            "role_label": role_lbl,
            "skill": skill_uri,
            "skill_label": skill_lbl,
        })

    return output


def get_low_confidence_relations(graph: Graph, threshold: float = 0.70) -> list[dict[str, Any]]:
    """
    Query reified rdf:Statement entries where cv:confidence < threshold to flag extraction ambiguities.
    Returns:
        list of dicts with keys: 'stmt', 'subject', 'predicate', 'object', 'confidence', 'source_snippet'
    """
    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX cv: <http://recruitment.org/cv#>

    SELECT DISTINCT ?stmt ?subject ?predicate ?object ?confidence ?sourceSnippet WHERE {{
        ?stmt rdf:type rdf:Statement ;
              rdf:subject ?subject ;
              rdf:predicate ?predicate ;
              rdf:object ?object ;
              cv:confidence ?confidence .
        OPTIONAL {{ ?stmt cv:sourceSnippet ?sourceSnippet }}
        FILTER(xsd:float(?confidence) < {float(threshold)})
    }}
    """

    try:
        results = graph.query(sparql_query)
    except Exception:
        # Fallback without xsd:float if datatype issue
        fallback_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX cv: <http://recruitment.org/cv#>

        SELECT DISTINCT ?stmt ?subject ?predicate ?object ?confidence ?sourceSnippet WHERE {{
            ?stmt rdf:type rdf:Statement ;
                  rdf:subject ?subject ;
                  rdf:predicate ?predicate ;
                  rdf:object ?object ;
                  cv:confidence ?confidence .
            OPTIONAL {{ ?stmt cv:sourceSnippet ?sourceSnippet }}
        }}
        """
        raw_results = graph.query(fallback_query)
        output: list[dict[str, Any]] = []
        for row in raw_results:
            conf_val = float(row.confidence) if row.confidence else 1.0
            if conf_val < threshold:
                output.append({
                    "stmt": str(row.stmt) if row.stmt else "",
                    "subject": str(row.subject) if row.subject else "",
                    "predicate": str(row.predicate) if row.predicate else "",
                    "object": str(row.object) if row.object else "",
                    "confidence": conf_val,
                    "source_snippet": str(row.sourceSnippet or ""),
                })
        return output

    output: list[dict[str, Any]] = []
    for row in results:
        conf_val = float(row.confidence) if row.confidence else 0.0
        output.append({
            "stmt": str(row.stmt) if row.stmt else "",
            "subject": str(row.subject) if row.subject else "",
            "predicate": str(row.predicate) if row.predicate else "",
            "object": str(row.object) if row.object else "",
            "confidence": conf_val,
            "source_snippet": str(row.sourceSnippet or ""),
        })

    return output


def match_skills_with_taxonomy(graph: Graph, target_parent_skill: str) -> list[dict[str, Any]]:
    """
    Use transitive property paths (skos:broader*) to find all leaf skills
    in the graph rolling up to a required high-level parent category.
    Returns:
        list of dicts with keys: 'skill', 'skill_label', 'parent', 'parent_label'
    """
    # Clean target parent name (e.g. 'skill:machinelearning' -> 'machinelearning')
    clean_target = target_parent_skill.strip().lower()
    if clean_target.startswith("skill:"):
        clean_target = clean_target[6:]
    elif clean_target.startswith("domain:"):
        clean_target = clean_target[7:]
    clean_target = re.sub(r"[\s\-_]+", "", clean_target)

    safe_target = clean_target.replace("\\", "\\\\").replace('"', '\\"')

    sparql_query = f"""
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX skill: <http://recruitment.org/skill#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?skill ?skillLabel ?parent ?parentLabel WHERE {{
        {{
            ?candidate (cv:usesSkill | cv:hasExperience/cv:usesSkill | cv:heldRole/cv:usesSkill) ?skill .
        }} UNION {{
            ?any (cv:usesSkill | cv:usedSkill) ?skill .
        }} UNION {{
            ?skill a cv:Skill .
        }}
        ?skill skos:broader* ?parent .
        OPTIONAL {{ ?skill rdfs:label ?skillLabel }}
        OPTIONAL {{ ?parent rdfs:label ?parentLabel }}
        FILTER(
            REGEX(STR(?parent), "{safe_target}", "i") ||
            REGEX(STR(?parentLabel), "{safe_target}", "i")
        )
    }}
    """

    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []

    for row in results:
        skill_uri = str(row.skill) if row.skill else ""
        parent_uri = str(row.parent) if row.parent else ""
        skill_lbl = str(row.skillLabel) if row.skillLabel else skill_uri.split("#")[-1]
        parent_lbl = str(row.parentLabel) if row.parentLabel else parent_uri.split("#")[-1]

        output.append({
            "skill": skill_uri,
            "skill_label": skill_lbl,
            "parent": parent_uri,
            "parent_label": parent_lbl,
        })

    return output


def get_document_dependencies(graph: Graph) -> list[dict[str, Any]]:
    """
    Traverse cv:dependsOn to list all dependencies between concepts, protocols, and specifications.
    """
    sparql_query = """
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?concept ?conceptLabel ?specification ?specificationLabel WHERE {
        ?concept cv:dependsOn ?specification .
        OPTIONAL { ?concept rdfs:label ?conceptLabel }
        OPTIONAL { ?specification rdfs:label ?specificationLabel }
    }
    """
    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []
    for row in results:
        c_uri = str(row.concept) if row.concept else ""
        s_uri = str(row.specification) if row.specification else ""
        c_lbl = str(row.conceptLabel) if row.conceptLabel else c_uri.split("#")[-1].replace("_", " ")
        s_lbl = str(row.specificationLabel) if row.specificationLabel else s_uri.split("#")[-1].replace("_", " ")
        output.append({
            "concept": c_uri,
            "concept_label": c_lbl,
            "specification": s_uri,
            "specification_label": s_lbl,
        })
    return output


def get_protocol_conflicts(graph: Graph) -> list[dict[str, Any]]:
    """
    Query cv:conflictsWith to retrieve protocol and specification incompatibilities.
    """
    sparql_query = """
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?p1 ?p1Label ?p2 ?p2Label WHERE {
        ?p1 cv:conflictsWith ?p2 .
        OPTIONAL { ?p1 rdfs:label ?p1Label }
        OPTIONAL { ?p2 rdfs:label ?p2Label }
    }
    """
    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []
    for row in results:
        p1_uri = str(row.p1) if row.p1 else ""
        p2_uri = str(row.p2) if row.p2 else ""
        p1_lbl = str(row.p1Label) if row.p1Label else p1_uri.split("#")[-1].replace("_", " ")
        p2_lbl = str(row.p2Label) if row.p2Label else p2_uri.split("#")[-1].replace("_", " ")
        output.append({
            "protocol_a": p1_uri,
            "protocol_a_label": p1_lbl,
            "protocol_b": p2_uri,
            "protocol_b_label": p2_lbl,
        })
    return output


def get_defined_parameters(graph: Graph) -> list[dict[str, Any]]:
    """
    Query cv:defines to list parameters declared across document sections.
    """
    sparql_query = """
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?section ?sectionLabel ?parameter ?parameterLabel WHERE {
        ?section cv:defines ?parameter .
        OPTIONAL { ?section rdfs:label ?sectionLabel }
        OPTIONAL { ?parameter rdfs:label ?parameterLabel }
    }
    """
    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []
    for row in results:
        sec_uri = str(row.section) if row.section else ""
        par_uri = str(row.parameter) if row.parameter else ""
        sec_lbl = str(row.sectionLabel) if row.sectionLabel else sec_uri.split("#")[-1].replace("_", " ")
        par_lbl = str(row.parameterLabel) if row.parameterLabel else par_uri.split("#")[-1].replace("_", " ")
        output.append({
            "section": sec_uri,
            "section_label": sec_lbl,
            "parameter": par_uri,
            "parameter_label": par_lbl,
        })
    return output


def find_circular_constraints(graph: Graph) -> list[dict[str, Any]]:
    """
    Use transitive SPARQL property paths (cv:dependsOn+) to detect circular constraints.
    """
    sparql_query = """
    PREFIX cv: <http://recruitment.org/cv#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?a ?aLabel ?b ?bLabel WHERE {
        ?a cv:dependsOn+ ?b .
        ?b cv:dependsOn+ ?a .
        FILTER(?a != ?b)
        OPTIONAL { ?a rdfs:label ?aLabel }
        OPTIONAL { ?b rdfs:label ?bLabel }
    }
    """
    results = graph.query(sparql_query)
    output: list[dict[str, Any]] = []
    for row in results:
        a_uri = str(row.a) if row.a else ""
        b_uri = str(row.b) if row.b else ""
        a_lbl = str(row.aLabel) if row.aLabel else a_uri.split("#")[-1].replace("_", " ")
        b_lbl = str(row.bLabel) if row.bLabel else b_uri.split("#")[-1].replace("_", " ")
        output.append({
            "entity_a": a_uri,
            "entity_a_label": a_lbl,
            "entity_b": b_uri,
            "entity_b_label": b_lbl,
        })
    return output


