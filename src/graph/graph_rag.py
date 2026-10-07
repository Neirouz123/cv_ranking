"""
src/graph/graph_rag.py
----------------------
GraphRAG community detection and qualitative fit summarization.
Partitions candidate subgraphs into modular communities using Louvain clustering,
traverses local and global graph topologies, and synthesizes explainable candidate critiques.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import networkx as nx
from networkx.algorithms.community import louvain_communities

from src.core.models import (
    CandidateGraphPayload,
    CandidateRankingResult,
    ConflictSeverity,
    JobDescriptionGraphPayload,
)
from src.core.ontology import (
    SKOS_SKILL_RELATIONS,
    calculate_taxonomic_distance,
    get_taxonomy_networkx,
    normalize_skill_name,
)
from src.core.validator import ConsistencyValidator
from src.graph.builder import GraphBuilder
from src.graph.matching import TopologicalMatcher
from src.pipeline.ranker import HybridRanker


class GraphRAGSummaryEngine:
    """
    Engine for community clustering, topological traversal,
    and narrative critique generation for candidates.
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self.builder = GraphBuilder(include_ontology_taxonomies=True)
        self.matcher = TopologicalMatcher()
        self.validator = ConsistencyValidator()
        self.ranker = HybridRanker(topological_matcher=self.matcher, validator=self.validator)
        self.tax_graph = get_taxonomy_networkx()

        self._llm = None
        api_key = os.environ.get("GROQ_API_KEY")
        if self.use_llm and api_key:
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.2)
            except Exception:
                self._llm = None

    def detect_communities(
        self, candidate_payload: CandidateGraphPayload
    ) -> list[dict[str, Any]]:
        """
        Partition the candidate knowledge graph into modules using Louvain community clustering.
        Returns cluster descriptions with member entities and domain labels.
        """
        _, multi_g = self.builder.build_candidate_graph(candidate_payload)

        # Convert to undirected simple graph for Louvain clustering
        undirected_g = nx.Graph()
        for u, v, data in multi_g.edges(data=True):
            weight = float(data.get("weight", 1.0))
            if undirected_g.has_edge(u, v):
                undirected_g[u][v]["weight"] += weight
            else:
                undirected_g.add_edge(u, v, weight=weight)

        for n, data in multi_g.nodes(data=True):
            if n not in undirected_g:
                undirected_g.add_node(n, **data)
            else:
                undirected_g.nodes[n].update(data)

        if undirected_g.number_of_nodes() == 0:
            return []

        # Louvain partition
        try:
            clusters = list(louvain_communities(undirected_g, weight="weight", seed=42))
        except Exception:
            # Fallback to connected components
            clusters = list(nx.connected_components(undirected_g))

        summaries: list[dict[str, Any]] = []
        for idx, cluster_nodes in enumerate(clusters):
            nodes_data = [
                multi_g.nodes[n] for n in cluster_nodes if n in multi_g.nodes
            ]

            skills = [
                n.get("label", n.get("id"))
                for n in nodes_data
                if n.get("category") == "SKILL"
            ]
            roles = [
                n.get("label", n.get("id"))
                for n in nodes_data
                if n.get("category") == "ROLE"
            ]
            companies = [
                n.get("label", n.get("id"))
                for n in nodes_data
                if n.get("category") == "COMPANY"
            ]

            cluster_title = self._label_cluster(skills, roles, idx)

            summaries.append({
                "cluster_id": idx + 1,
                "title": cluster_title,
                "size": len(cluster_nodes),
                "skills": skills,
                "roles": roles,
                "companies": companies,
                "node_ids": list(cluster_nodes),
            })

        return summaries

    def _label_cluster(self, skills: list[str], roles: list[str], idx: int) -> str:
        """Heuristically assign a semantic domain label to a cluster."""
        combined_text = " ".join(skills + roles).lower()

        if any(w in combined_text for w in ["pytorch", "tensorflow", "vision", "nlp", "learning", "data", "scikit"]):
            return "AI, Machine Learning & Data Science"
        if any(w in combined_text for w in ["react", "vue", "angular", "frontend", "javascript", "css"]):
            return "Frontend & Web User Interfaces"
        if any(w in combined_text for w in ["fastapi", "django", "flask", "springboot", "java", "backend", "python"]):
            return "Backend Architecture & APIs"
        if any(w in combined_text for w in ["docker", "kubernetes", "cloud", "aws", "gcp", "azure", "cicd"]):
            return "Cloud, DevOps & Infrastructure"
        if any(w in combined_text for w in ["sql", "postgres", "kafka", "spark", "nosql", "redis"]):
            return "Data Infrastructure & Streaming"

        if roles:
            return f"Experience Cluster ({roles[0]})"
        return f"Specialization Module {idx + 1}"

    def generate_community_prompts(
        self, candidate_payload: CandidateGraphPayload
    ) -> list[dict[str, str]]:
        """
        Generate prompt inputs per community cluster (e.g. 'Cluster 1: Embedded IoT', 'Cluster 2: Machine Learning')
        to feed an LLM synthesis step explaining candidate strengths and gaps.
        """
        clusters = self.detect_communities(candidate_payload)
        prompts: list[dict[str, str]] = []

        for c in clusters:
            skills_str = ", ".join(c["skills"]) if c["skills"] else "None specified"
            roles_str = ", ".join(c["roles"]) if c["roles"] else "General experience"
            companies_str = ", ".join(c["companies"]) if c["companies"] else "Unspecified"

            prompt_text = (
                f"Cluster {c['cluster_id']}: {c['title']}\n"
                f"- Extracted Tools & Skills: {skills_str}\n"
                f"- Associated Professional Roles: {roles_str}\n"
                f"- Organizations: {companies_str}\n"
                f"Synthesize the candidate's competence, strengths, and technical gaps in this domain."
            )
            prompts.append({
                "cluster_id": str(c["cluster_id"]),
                "cluster_title": f"Cluster {c['cluster_id']}: {c['title']}",
                "prompt_input": prompt_text,
            })

        return prompts

    def generate_candidate_critique(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
    ) -> str:
        """
        Produce an explainable audit report detailing:
        - Matching skills (exact vs inferred taxonomy)
        - Career trajectory continuity
        - Unresolved discrepancies / temporal warnings
        - Community cluster breakdown
        """
        # Run ranking and topological matching
        ranking_result = self.ranker.score_candidate(candidate_payload, jd_payload)
        breakdown = ranking_result.graph_breakdown
        val_report = ranking_result.validation_report
        clusters = self.detect_communities(candidate_payload)

        # Career trajectory continuity analysis
        trajectory = self._analyze_career_trajectory(candidate_payload)

        critique_md = self._render_critique_markdown(
            candidate_payload=candidate_payload,
            jd_payload=jd_payload,
            ranking_result=ranking_result,
            breakdown=breakdown,
            val_report=val_report,
            clusters=clusters,
            trajectory=trajectory,
        )

        return critique_md

    def _analyze_career_trajectory(self, candidate_payload: CandidateGraphPayload) -> dict[str, Any]:
        """Examine role progression and chronology for career continuity."""
        roles_held: list[dict[str, Any]] = []

        for rel in candidate_payload.relations:
            if rel.predicate.value in {"HELD_ROLE", "WORKED_AT"}:
                role_entity = candidate_payload.get_entity(rel.object_id) or candidate_payload.get_entity(rel.subject_id)
                label = role_entity.label if role_entity else rel.object_id
                roles_held.append({
                    "role": label,
                    "start": rel.start_date or "Unknown",
                    "end": rel.end_date or "Present",
                })

        return {
            "num_positions": len(roles_held),
            "positions": roles_held,
            "has_continuous_path": len(roles_held) > 0,
        }

    def _render_critique_markdown(
        self,
        candidate_payload: CandidateGraphPayload,
        jd_payload: JobDescriptionGraphPayload,
        ranking_result: CandidateRankingResult,
        breakdown: Any,
        val_report: Any,
        clusters: list[dict[str, Any]],
        trajectory: dict[str, Any],
    ) -> str:
        """Render detailed audit report in Markdown format."""
        score = ranking_result.final_score
        status_badge = "🟢 Strong Fit" if score >= 75 else ("🟠 Moderate Fit" if score >= 50 else "🔴 Weak Fit")

        sections = [
            f"# 🎯 Candidate Fit Audit: {candidate_payload.name}",
            f"**Job Reference:** {jd_payload.title}  ",
            f"**Overall Assessment:** {status_badge} (**{score}/100**)  ",
            f"- **Graph Match Score:** {ranking_result.graph_match_score:.2f} (PPR: {breakdown.ppr_score:.2f}, SKOS Taxonomy: {breakdown.ontological_distance_score:.2f})",
            f"- **Text Relevance (TF-IDF):** {ranking_result.vector_similarity:.2f}",
            f"- **Conflict Penalty Factor:** {ranking_result.conflict_penalty:.2f}",
            "\n---",
            "\n## 1. Skill Alignment & Ontological Match",
        ]

        # Exact Matches
        if breakdown.matched_skills_exact:
            sections.append(f"### Direct Matches ({len(breakdown.matched_skills_exact)})")
            for skill in breakdown.matched_skills_exact:
                sections.append(f"- ✅ **{skill}** (Direct Match)")
        else:
            sections.append("- ⚠️ No direct exact skill matches found.")

        # Inferred Matches via SKOS Taxonomy
        if breakdown.matched_skills_inferred:
            sections.append(f"\n### Ontologically Inferred Matches ({len(breakdown.matched_skills_inferred)})")
            sections.append("Partial credit awarded based on SKOS taxonomic proximity:")
            for item in breakdown.matched_skills_inferred:
                sections.append(
                    f"- 🔄 **Candidate Skill:** `{item['candidate_skill']}` ➔ "
                    f"**Required:** `{item['jd_skill']}` "
                    f"(Taxpath hops: {int(item['distance'])}, Proximity similarity: {item['similarity']:.2f})"
                )

        # Community Clusters
        sections.append("\n## 2. GraphRAG Knowledge Communities")
        sections.append("Candidate graph was clustered into the following modular skill domains:")
        for cluster in clusters:
            skills_str = ", ".join(cluster["skills"][:6]) if cluster["skills"] else "None explicitly isolated"
            sections.append(f"- **Module {cluster['cluster_id']}: {cluster['title']}**")
            sections.append(f"  - Key technologies: {skills_str}")
            if cluster["roles"]:
                sections.append(f"  - Associated roles: {', '.join(cluster['roles'][:3])}")

        # Career Trajectory Continuity
        sections.append("\n## 3. Career Trajectory & Continuity")
        if trajectory["positions"]:
            for pos in trajectory["positions"]:
                sections.append(f"- 📌 **{pos['role']}** ({pos['start']} ➔ {pos['end']})")
        else:
            sections.append("- *No discrete chronological positions extracted.*")

        # Temporal & Contradiction Warnings
        sections.append("\n## 4. Integrity, Temporal & Contradiction Warnings")
        if val_report.conflicts:
            sections.append(f"⚠️ **{len(val_report.conflicts)} Warning(s) or Contradiction(s) Detected:**")
            for conf in val_report.conflicts:
                sev_icon = "🛑" if conf.severity == ConflictSeverity.HIGH else "⚠️"
                sections.append(f"- {sev_icon} **[{conf.conflict_type.value}]** {conf.message}")
        else:
            sections.append("✅ **No contradictions, temporal inversions, or anachronisms detected.** Candidate profile is self-consistent.")

        sections.append("\n---")
        sections.append("*Generated by CV-Ranker Knowledge Graph & GraphRAG Engine.*")

        return "\n".join(sections)

