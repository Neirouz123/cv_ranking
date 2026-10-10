"""
src/graph/matching.py
---------------------
Topological graph matching engine between Candidates and Job Descriptions.
Implements:
1. Personalized PageRank (PPR) seeded on JD requirements.
2. Ontological taxpath distance scoring via SKOS hierarchy.
3. Jaccard similarity of skill/entity sets.
"""

from __future__ import annotations

import networkx as nx
from typing import Optional

from src.core.models import (
    CandidateGraphPayload,
    GraphMatchBreakdown,
    JobDescriptionGraphPayload,
)
from src.core.ontology import (
    calculate_taxonomic_distance,
    get_taxonomy_networkx,
    normalize_skill_name,
    taxonomic_similarity,
)
from src.graph.builder import GraphBuilder


class TopologicalMatcher:
    """
    Computes graph-based match metrics between candidate knowledge subgraphs
    and job descriptions.
    """

    def __init__(
        self,
        weight_ontology: float = 0.50,
        weight_ppr: float = 0.35,
        weight_jaccard: float = 0.15,
        ppr_alpha: float = 0.85,
        use_taxonomy: bool = True,
    ) -> None:
        self.w_onto = weight_ontology
        self.w_ppr = weight_ppr
        self.w_jaccard = weight_jaccard
        self.ppr_alpha = ppr_alpha
        self.use_taxonomy = use_taxonomy
        self.builder = GraphBuilder(include_ontology_taxonomies=True)
        self.tax_graph = get_taxonomy_networkx()

    def compute_match(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
    ) -> GraphMatchBreakdown:
        """
        Compute full topological matching breakdown between candidate and job description,
        applying Must-have vs Nice-to-have weights and identifying missing mandatory skills.
        """
        # Extract normalized skill sets
        cand_skills = {
            normalize_skill_name(e.id): e.label
            for e in candidate_payload.get_skills()
        }
        jd_skills = {
            normalize_skill_name(e.id): e.label
            for e in jd_payload.get_skills()
        }

        # Resolve Must vs Nice skills from JD payload
        must_skills_set = set(normalize_skill_name(s) for s in jd_payload.get_must_skills())

        # 1. Jaccard similarity on exact skill sets
        cand_set = set(cand_skills.keys())
        jd_set = set(jd_skills.keys())

        if not jd_set and not cand_set:
            jaccard = 1.0
        elif not jd_set or not cand_set:
            jaccard = 0.0
        else:
            jaccard = len(cand_set & jd_set) / float(len(cand_set | jd_set))

        # 2. Ontological distance scoring via SKOS taxonomy
        matched_exact: list[str] = []
        matched_inferred: list[dict] = []
        weighted_scores: list[float] = []
        total_weights: list[float] = []
        missing_must_skills: list[str] = []

        if not jd_set:
            onto_score = 1.0 if cand_set else 0.5
        else:
            for jd_s_id, jd_label in jd_skills.items():
                is_must = (jd_s_id in must_skills_set) or (not must_skills_set)
                req_weight = 1.0 if is_must else 0.5
                total_weights.append(req_weight)

                if jd_s_id in cand_set:
                    matched_exact.append(jd_skills[jd_s_id])
                    weighted_scores.append(1.0 * req_weight)
                else:
                    best_sim = 0.0
                    best_cand_s = None
                    best_dist = float("inf")

                    if self.use_taxonomy:
                        # Find highest taxonomical similarity among candidate skills
                        for c_s_id, c_label in cand_skills.items():
                            dist = calculate_taxonomic_distance(c_s_id, jd_s_id, self.tax_graph)
                            sim = taxonomic_similarity(c_s_id, jd_s_id, taxonomy_graph=self.tax_graph)
                            if sim > best_sim:
                                best_sim = sim
                                best_cand_s = c_label
                                best_dist = dist

                    weighted_scores.append(best_sim * req_weight)
                    if best_sim > 0.0 and best_cand_s is not None:
                        matched_inferred.append({
                            "candidate_skill": best_cand_s,
                            "jd_skill": jd_label,
                            "distance": best_dist,
                            "similarity": round(best_sim, 3),
                        })
                    elif is_must and best_sim == 0.0:
                        missing_must_skills.append(jd_label)

            sum_w = sum(total_weights) if total_weights else 1.0
            onto_score = sum(weighted_scores) / float(sum_w)

        # 3. Personalized PageRank (PPR)
        if self.w_ppr > 0.0:
            ppr_score = self._compute_ppr(candidate_payload, jd_payload, jd_set)
        else:
            ppr_score = 0.0

        # 4. Synthesize final graph match score with dynamic normalization
        w_sum = self.w_onto + self.w_ppr + self.w_jaccard
        if w_sum > 0:
            raw_graph_score = (
                self.w_onto * onto_score
                + self.w_ppr * ppr_score
                + self.w_jaccard * jaccard
            ) / w_sum
        else:
            raw_graph_score = onto_score

        graph_match_score = min(1.0, max(0.0, raw_graph_score))

        return GraphMatchBreakdown(
            ppr_score=round(ppr_score, 4),
            ontological_distance_score=round(onto_score, 4),
            jaccard_similarity=round(jaccard, 4),
            graph_match_score=round(graph_match_score, 4),
            matched_skills_exact=matched_exact,
            matched_skills_inferred=matched_inferred,
            missing_must_skills=missing_must_skills,
        )

    def _compute_ppr(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
        jd_skill_set: set[str],
    ) -> float:
        """
        Compute Personalized PageRank on a joint candidate + taxonomy graph.
        Diffusion seeds are placed uniformly on the JD-required skills.
        The score measures the probability density accumulated on the candidate node
        and candidate-specific skills.
        """
        if not jd_skill_set:
            return 0.5

        # Build joint graph: candidate relations + SKOS taxonomy edges
        joint_graph = nx.Graph()

        # Add candidate network edges
        _, cand_nx = self.builder.build_candidate_graph(candidate_payload)
        for u, v, data in cand_nx.edges(data=True):
            joint_graph.add_edge(u, v, weight=data.get("weight", 1.0))

        # Add taxonomy edges so diffusion flows between related concepts
        for u, v, data in self.tax_graph.edges(data=True):
            joint_graph.add_edge(u, v, weight=0.8)

        # Ensure all JD skills are in joint graph
        for s in jd_skill_set:
            if s not in joint_graph:
                joint_graph.add_node(s)

        # If graph is empty or disconnected without edges
        if joint_graph.number_of_nodes() == 0:
            return 0.0

        # Construct personalization vector on JD skills
        personalization = {node: 0.0 for node in joint_graph.nodes()}
        seeds = [s for s in jd_skill_set if s in personalization]

        if not seeds:
            return 0.0

        uniform_seed = 1.0 / len(seeds)
        for s in seeds:
            personalization[s] = uniform_seed

        try:
            pagerank_dist = nx.pagerank(
                joint_graph,
                alpha=self.ppr_alpha,
                personalization=personalization,
                max_iter=100,
                tol=1e-5,
            )
        except Exception:
            return 0.0

        cand_node_id = candidate_payload.candidate_id.strip().lower()
        cand_score = pagerank_dist.get(cand_node_id, 0.0)

        # Also sum diffusion on candidate's direct skills
        cand_skill_scores = [
            pagerank_dist.get(normalize_skill_name(e.id), 0.0)
            for e in candidate_payload.get_skills()
        ]
        sum_skills = sum(cand_skill_scores)

        # Scale by node count to make PPR scale-independent with an upper bound
        node_count = joint_graph.number_of_nodes()
        scale_factor = min(2.5, max(1.0, node_count / 10.0))
        effective_ppr = (cand_score * 2.0 + sum_skills) * scale_factor

        # Normalize score into [0.0, 1.0]
        normalized_ppr = min(1.0, max(0.0, effective_ppr))
        return normalized_ppr

