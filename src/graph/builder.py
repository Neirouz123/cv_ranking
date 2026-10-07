"""
src/graph/builder.py
--------------------
Graph ingestion and bi-directional conversion between RDFLib semantic graphs
and NetworkX directed multigraphs (nx.MultiDiGraph), preserving edge attributes
(dates, confidence, predicate types).
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import networkx as nx
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, XSD

from src.core.models import (
    CandidateGraphPayload,
    Entity,
    EntityCategory,
    JobDescriptionGraphPayload,
    PredicateType,
    Relation,
)
from src.core.ontology import CV, build_skos_ontology, normalize_skill_name

# Mapping from PredicateType enum to RDFLib Property
PREDICATE_TO_RDF: dict[PredicateType, URIRef] = {
    PredicateType.HELD_ROLE: CV.heldRole,
    PredicateType.WORKED_AT: CV.workedAt,
    PredicateType.USED_SKILL: CV.usedSkill,
    PredicateType.DELIVERED_PROJECT: CV.deliveredProject,
    PredicateType.EARNED_DEGREE: CV.earnedDegree,
}

RDF_TO_PREDICATE: dict[URIRef, PredicateType] = {
    CV.heldRole: PredicateType.HELD_ROLE,
    CV.workedAt: PredicateType.WORKED_AT,
    CV.usedSkill: PredicateType.USED_SKILL,
    CV.deliveredProject: PredicateType.DELIVERED_PROJECT,
    CV.earnedDegree: PredicateType.EARNED_DEGREE,
}

CATEGORY_TO_RDF_CLASS: dict[EntityCategory, URIRef] = {
    EntityCategory.CANDIDATE: CV.Candidate,
    EntityCategory.ROLE: CV.Role,
    EntityCategory.SKILL: CV.Skill,
    EntityCategory.DEGREE: CV.Degree,
    EntityCategory.COMPANY: CV.Company,
    EntityCategory.PROJECT: CV.Project,
}


def entity_id_to_uri(entity_id: str) -> URIRef:
    """Deterministic URI creation from an entity identifier."""
    clean_id = entity_id.strip().lower().replace(":", "_").replace(" ", "_")
    return CV[clean_id]


def uri_to_entity_id(uri: URIRef) -> str:
    """Recover entity ID from an RDF URI."""
    suffix = str(uri).split("#")[-1]
    if "_" in suffix:
        prefix, *rest = suffix.split("_")
        return f"{prefix}:{'_'.join(rest)}"
    return suffix


class GraphBuilder:
    """
    Handles ingestion of candidate payloads into RDFLib and NetworkX,
    as well as bidirectional translation between RDF and graph representations.
    """

    def __init__(self, include_ontology_taxonomies: bool = True) -> None:
        self.include_ontology = include_ontology_taxonomies

    def payload_to_rdflib(
        self,
        payload: CandidateGraphPayload | JobDescriptionGraphPayload,
        base_graph: Optional[Graph] = None,
    ) -> Graph:
        """
        Ingest an extracted candidate or job payload into an RDFLib graph.
        Preserves temporal qualifiers and confidence via RDF statement reification.
        """
        g = base_graph if base_graph is not None else (build_skos_ontology() if self.include_ontology else Graph())
        g.bind("cv", CV)
        g.bind("skos", SKOS)
        g.bind("rdf", RDF)

        # 1. Ingest Entities
        for entity in payload.entities:
            e_uri = entity_id_to_uri(entity.id)
            rdf_class = CATEGORY_TO_RDF_CLASS.get(entity.category, CV.Entity)

            g.add((e_uri, RDF.type, rdf_class))
            g.add((e_uri, RDFS.label, Literal(entity.label, datatype=XSD.string)))
            g.add((e_uri, CV.entityId, Literal(entity.id, datatype=XSD.string)))
            g.add((e_uri, CV.category, Literal(entity.category.value, datatype=XSD.string)))

            if entity.claimed_years is not None:
                g.add((e_uri, CV.claimedYears, Literal(float(entity.claimed_years), datatype=XSD.float)))

        # If payload is CandidateGraphPayload, also assert candidate node if not already present
        if isinstance(payload, CandidateGraphPayload):
            c_uri = entity_id_to_uri(payload.candidate_id)
            g.add((c_uri, RDF.type, CV.Candidate))
            g.add((c_uri, RDFS.label, Literal(payload.name, datatype=XSD.string)))
            g.add((c_uri, CV.entityId, Literal(payload.candidate_id, datatype=XSD.string)))

        # 2. Ingest Relations
        for idx, rel in enumerate(payload.relations):
            s_uri = entity_id_to_uri(rel.subject_id)
            o_uri = entity_id_to_uri(rel.object_id)
            p_uri = PREDICATE_TO_RDF.get(rel.predicate, CV[rel.predicate.value])

            # Direct triple
            g.add((s_uri, p_uri, o_uri))

            # Reified Statement node to preserve edge attributes (dates, confidence)
            stmt_id = f"stmt_{uuid.uuid4().hex[:8]}"
            stmt_uri = CV[stmt_id]
            g.add((stmt_uri, RDF.type, RDF.Statement))
            g.add((stmt_uri, RDF.subject, s_uri))
            g.add((stmt_uri, RDF.predicate, p_uri))
            g.add((stmt_uri, RDF.object, o_uri))
            g.add((stmt_uri, CV.subjectId, Literal(rel.subject_id, datatype=XSD.string)))
            g.add((stmt_uri, CV.objectId, Literal(rel.object_id, datatype=XSD.string)))
            g.add((stmt_uri, CV.predicateType, Literal(rel.predicate.value, datatype=XSD.string)))
            g.add((stmt_uri, CV.confidence, Literal(float(rel.confidence), datatype=XSD.float)))

            if rel.start_date:
                g.add((stmt_uri, CV.startDate, Literal(rel.start_date, datatype=XSD.string)))
            if rel.end_date:
                g.add((stmt_uri, CV.endDate, Literal(rel.end_date, datatype=XSD.string)))

        return g

    def rdflib_to_networkx(self, g: Graph) -> nx.MultiDiGraph:
        """
        Convert an RDFLib graph into a directed NetworkX multigraph (nx.MultiDiGraph).
        Preserves node attributes and all edge attributes (dates, confidence, predicate type).
        """
        nx_graph = nx.MultiDiGraph()

        def _get_node_id(node_uri: URIRef) -> str:
            for eid in g.objects(node_uri, CV.entityId):
                return str(eid)
            suffix = str(node_uri).split("#")[-1]
            return suffix

        # 1. Collect entities and node attributes
        for s in g.subjects(RDF.type, None):
            if isinstance(s, URIRef) and not str(s).startswith(str(CV) + "stmt_"):
                node_id = _get_node_id(s)

                label = None
                for l in g.objects(s, RDFS.label):
                    label = str(l)
                    break
                if not label:
                    label = node_id

                category = None
                for c in g.objects(s, CV.category):
                    category = str(c)
                    break
                if not category:
                    for t in g.objects(s, RDF.type):
                        t_suffix = str(t).split("#")[-1].upper()
                        if t_suffix in EntityCategory.__members__:
                            category = t_suffix
                            break

                claimed_years = None
                for cy in g.objects(s, CV.claimedYears):
                    try:
                        claimed_years = float(cy)
                    except (ValueError, TypeError):
                        pass

                nx_graph.add_node(
                    node_id,
                    id=node_id,
                    label=label,
                    category=category or "UNKNOWN",
                    uri=str(s),
                    claimed_years=claimed_years,
                )

        # 2. Extract edge reifications (statements with attributes)
        reified_edges: dict[Any, dict[str, Any]] = {}
        for stmt in g.subjects(RDF.type, RDF.Statement):
            s_val = next(g.objects(stmt, RDF.subject), None)
            p_val = next(g.objects(stmt, RDF.predicate), None)
            o_val = next(g.objects(stmt, RDF.object), None)

            if s_val and p_val and o_val:
                s_id = next((str(x) for x in g.objects(stmt, CV.subjectId)), None) or _get_node_id(s_val)
                o_id = next((str(x) for x in g.objects(stmt, CV.objectId)), None) or _get_node_id(o_val)
                p_name = str(p_val).split("#")[-1]

                start_date = None
                end_date = None
                confidence = 1.0
                pred_type = p_name

                for d in g.objects(stmt, CV.startDate):
                    start_date = str(d)
                for d in g.objects(stmt, CV.endDate):
                    end_date = str(d)
                for c in g.objects(stmt, CV.confidence):
                    try:
                        confidence = float(c)
                    except (ValueError, TypeError):
                        pass
                for pt in g.objects(stmt, CV.predicateType):
                    pred_type = str(pt)

                edge_info = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "confidence": confidence,
                    "predicate_type": pred_type,
                }
                # Store multiple index keys for resilient matching
                reified_edges[(s_id, o_id, pred_type.upper())] = edge_info
                reified_edges[(s_id, o_id, p_name)] = edge_info
                reified_edges[(s_id, o_id, p_name.lower())] = edge_info
                reified_edges[(s_id, o_id)] = edge_info

        # 3. Add edges from RDF triples
        for s, p, o in g:
            if p in {RDF.type, RDFS.label, CV.entityId, CV.category, CV.claimedYears, SKOS.inScheme}:
                continue
            if isinstance(s, URIRef) and str(s).startswith(str(CV) + "stmt_"):
                continue
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue

            s_id = _get_node_id(s)
            o_id = _get_node_id(o)
            p_name = str(p).split("#")[-1]

            # Ensure nodes exist
            if s_id not in nx_graph:
                nx_graph.add_node(s_id, id=s_id, label=s_id, category="UNKNOWN", uri=str(s))
            if o_id not in nx_graph:
                nx_graph.add_node(o_id, id=o_id, label=o_id, category="UNKNOWN", uri=str(o))

            reified = (
                reified_edges.get((s_id, o_id, p_name))
                or reified_edges.get((s_id, o_id, p_name.lower()))
                or reified_edges.get((s_id, o_id))
                or {}
            )

            ptype = reified.get("predicate_type") or RDF_TO_PREDICATE.get(p, PredicateType.USED_SKILL).value

            nx_graph.add_edge(
                s_id,
                o_id,
                predicate=p_name,
                predicate_type=ptype,
                start_date=reified.get("start_date"),
                end_date=reified.get("end_date"),
                confidence=reified.get("confidence", 1.0),
                weight=float(reified.get("confidence", 1.0)),
            )

        return nx_graph

    def networkx_to_rdflib(self, nx_graph: nx.MultiDiGraph) -> Graph:
        """
        Convert a NetworkX MultiDiGraph back into an RDFLib Graph.
        Re-constructs statements, types, and edge attributes.
        """
        g = Graph()
        g.bind("cv", CV)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)

        # 1. Convert Nodes
        for node_id, data in nx_graph.nodes(data=True):
            n_uri = entity_id_to_uri(node_id)
            cat_str = data.get("category", "UNKNOWN").upper()
            if cat_str in EntityCategory.__members__:
                rdf_class = CATEGORY_TO_RDF_CLASS[EntityCategory[cat_str]]
            else:
                rdf_class = CV.Entity

            g.add((n_uri, RDF.type, rdf_class))
            g.add((n_uri, RDFS.label, Literal(data.get("label", node_id), datatype=XSD.string)))
            g.add((n_uri, CV.entityId, Literal(node_id, datatype=XSD.string)))

            if data.get("claimed_years") is not None:
                g.add((n_uri, CV.claimedYears, Literal(float(data["claimed_years"]), datatype=XSD.float)))

        # 2. Convert Edges
        for u, v, key, data in nx_graph.edges(keys=True, data=True):
            u_uri = entity_id_to_uri(u)
            v_uri = entity_id_to_uri(v)

            pred_str = data.get("predicate") or "usedSkill"
            pred_uri = CV[pred_str]
            g.add((u_uri, pred_uri, v_uri))

            # Reified statement for edge attributes
            stmt_id = f"stmt_{uuid.uuid4().hex[:8]}"
            stmt_uri = CV[stmt_id]
            g.add((stmt_uri, RDF.type, RDF.Statement))
            g.add((stmt_uri, RDF.subject, u_uri))
            g.add((stmt_uri, RDF.predicate, pred_uri))
            g.add((stmt_uri, RDF.object, v_uri))

            ptype = data.get("predicate_type", pred_str)
            g.add((stmt_uri, CV.predicateType, Literal(ptype, datatype=XSD.string)))

            conf = float(data.get("confidence", 1.0))
            g.add((stmt_uri, CV.confidence, Literal(conf, datatype=XSD.float)))

            if data.get("start_date"):
                g.add((stmt_uri, CV.startDate, Literal(data["start_date"], datatype=XSD.string)))
            if data.get("end_date"):
                g.add((stmt_uri, CV.endDate, Literal(data["end_date"], datatype=XSD.string)))

        return g

    def build_candidate_graph(
        self, payload: CandidateGraphPayload
    ) -> tuple[Graph, nx.MultiDiGraph]:
        """Convenience method returning both RDFLib and NetworkX graph representations."""
        rdf_g = self.payload_to_rdflib(payload)
        nx_g = self.rdflib_to_networkx(rdf_g)
        return rdf_g, nx_g
