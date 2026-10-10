"""
app.py
------
Streamlit interface for the Knowledge Graph, SPARQL Audit & GraphRAG CV Ranker,
with dual operating modes:
1. CV & Job Matching (Candidate vs Job Description)
2. Technical Document & Specification (RFCs, Papers, Protocols, Parameter Constraints)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.core.validator import validate_candidate_graph, validate_document_graph
from src.extraction.extractor import ExtractedProfile
from src.graph.builder import GraphBuilder
from src.graph.graph_rag import GraphRAGSummaryEngine
from src.graph.matching import TopologicalMatcher
from src.graph.pipeline import graph, kg_pipeline
from src.graph.sparql_queries import (
    find_circular_constraints,
    get_defined_parameters,
    get_document_dependencies,
    get_low_confidence_relations,
    get_protocol_conflicts,
    get_skills_for_role,
    match_skills_with_taxonomy,
)
from src.parsing.document_parser import DocumentParsingError, extract_text
from src.pipeline.extractor import KnowledgeGraphExtractor, get_llm_client
from src.pipeline.ranker import HybridRanker, RankingWeights, compute_final_rank, compute_vector_similarity
from src.scoring.scorer import ScoreBreakdown, ScoringWeights

st.set_page_config(
    page_title="Dossier RH & Technical KG Explorer",
    page_icon="🗂️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling — lightweight case-file theme
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #1B211F; }
    h1, h2, h3 { font-family: Georgia, serif; }
    .score-badge {
        display:inline-block; padding:6px 14px; border-radius:999px;
        font-family: monospace; font-weight:600; font-size:15px;
    }
    .rank-1 { background:#3F625922; border:1px solid #3F6259; color:#3F6259; }
    .rank-2 { background:#A9823D22; border:1px solid #A9823D; color:#A9823D; }
    .rank-3 { background:#A6433D22; border:1px solid #A6433D; color:#A6433D; }
    .conflict-card {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #2D2322;
        border-left: 4px solid #D9534F;
    }
    .clean-card {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #1E2B25;
        border-left: 4px solid #5CB85C;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — Operating Mode & LLM Toggle
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Mode & Options")

operating_mode = st.sidebar.radio(
    "Operating Mode",
    ["CV & Job Matching", "Technical Document & Specification"],
    index=0,
)

st.sidebar.divider()

st.sidebar.subheader("🤖 LLM Enrichment")
enable_llm = st.sidebar.toggle("Enable LLM Parsing (ChatGroq)", value=False)

env_key = os.environ.get("GROQ_API_KEY", "").strip()
api_key_input = ""

if enable_llm and not env_key:
    api_key_input = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Fournissez une clé API Groq pour activer l'analyse avancée LLaMA 3.3 70B.",
    )

effective_key = env_key or (api_key_input.strip() if api_key_input else None)
effective_use_llm = bool(enable_llm and effective_key)

if effective_use_llm:
    st.sidebar.markdown(
        '<div style="background-color:#1e3d2f; border:1px solid #2e7d32; color:#81c784; '
        'padding:6px 12px; border-radius:20px; font-size:12px; font-weight:bold; text-align:center;">'
        '🟢 LLM Active (ChatGroq)</div>',
        unsafe_allow_html=True,
    )
elif enable_llm and not effective_key:
    st.sidebar.markdown(
        '<div style="background-color:#3d351e; border:1px solid #f57f17; color:#ffb74d; '
        'padding:6px 12px; border-radius:20px; font-size:12px; font-weight:bold; text-align:center;">'
        '🟡 Rule-Based / Offline Mode (Clé requise)</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div style="background-color:#2a2f2d; border:1px solid #5a6460; color:#b0bec5; '
        'padding:6px 12px; border-radius:20px; font-size:12px; font-weight:bold; text-align:center;">'
        '🟡 Rule-Based / Offline Mode</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# MODE 1: CV & JOB MATCHING
# ==========================================================================
if operating_mode == "CV & Job Matching":
    st.sidebar.divider()
    st.sidebar.header("⚖️ Pondération Hybride")
    st.sidebar.caption("Ajustez l'importance des composantes du score hybride :")

    w_vec_slider = st.sidebar.slider("Similarité Textuelle (Vecteur / TF-IDF)", 0, 100, 25, help="Pondération alpha.")
    w_graph_slider = st.sidebar.slider("Match Topologique Graphe (PPR + SKOS)", 0, 100, 45, help="Pondération bêta.")
    w_exp_slider = st.sidebar.slider("Expérience & Séniorité (Durée)", 0, 100, 15, help="Pondération delta.")
    w_deg_slider = st.sidebar.slider("Niveau de Diplôme & Qualification", 0, 100, 15, help="Pondération epsilon.")
    w_pen_slider = st.sidebar.slider("Pénalité d'Intégrité / Conflits", 0, 100, 25, help="Pondération gamma déductive.")

    sum_pos = w_vec_slider + w_graph_slider + w_exp_slider + w_deg_slider
    if sum_pos == 0:
        st.sidebar.error("Au moins un critère de matching positif doit être supérieur à 0.")
        st.stop()

    pct_vec = (w_vec_slider / sum_pos) * 100
    pct_graph = (w_graph_slider / sum_pos) * 100
    pct_exp = (w_exp_slider / sum_pos) * 100
    pct_deg = (w_deg_slider / sum_pos) * 100

    st.sidebar.caption(
        f"**Pondérations effectives normalisées :**\n\n"
        f"• Vecteur : **{pct_vec:.1f}%**\n\n"
        f"• Graphe : **{pct_graph:.1f}%**\n\n"
        f"• Expérience : **{pct_exp:.1f}%**\n\n"
        f"• Diplôme : **{pct_deg:.1f}%**\n\n"
        f"*(Pénalité conflits : -{w_pen_slider}%)*"
    )

    ranking_weights = RankingWeights(
        alpha=w_vec_slider / 100.0,
        beta=w_graph_slider / 100.0,
        delta=w_exp_slider / 100.0,
        epsilon=w_deg_slider / 100.0,
        gamma=w_pen_slider / 100.0,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Guide de lecture des scores :**\n\n"
        "- 🟢 **≥ 75** : Forte correspondance & intégrité validée\n"
        "- 🟠 **50–74** : Correspondance partielle / compétences connexes\n"
        "- 🔴 **< 50** : Profil éloigné ou pénalisé par des contradictions"
    )

    st.title("🗂️ Dossier RH — Knowledge Graph & GraphRAG Ranker")
    st.caption(
        "Système d'évaluation de candidats fondé sur un **graphe de connaissances ontologique (RDFLib / SKOS)**, "
        "du **scoring topologique (NetworkX / Personalized PageRank)**, un **audit d'incohérences temporelles & factuelles**, "
        "et des **synthèses qualitatives GraphRAG**."
    )

    # Job offer input
    st.subheader("1. Offre d'emploi")
    job_input_mode = st.radio(
        "Méthode de saisie", ["Coller le texte", "Importer un fichier"], horizontal=True, key="job_mode"
    )

    job_text: str | None = None
    if job_input_mode == "Coller le texte":
        job_text = st.text_area(
            "Texte de l'offre d'emploi",
            height=160,
            placeholder="Collez ici la description complète du poste : missions, compétences requises, expérience, diplôme...",
        )
    else:
        job_file = st.file_uploader("Fichier de l'offre (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="job_file")
        if job_file is not None:
            try:
                job_text = extract_text(job_file, filename=job_file.name)
                with st.expander("Texte extrait de l'offre"):
                    st.text(job_text)
            except DocumentParsingError as exc:
                st.error(str(exc))

    # Résumés input
    st.subheader("2. CV des candidats")
    cv_files = st.file_uploader(
        "Importez un ou plusieurs CV (PDF, DOCX ou TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    run = st.button("🔎 Lancer le classement & l'audit", type="primary", use_container_width=False)

    if run:
        if not job_text or not job_text.strip():
            st.error("Merci de renseigner l'offre d'emploi (texte collé ou fichier importé).")
            st.stop()
        if not cv_files:
            st.error("Merci d'importer au moins un CV.")
            st.stop()

        extractor = KnowledgeGraphExtractor(use_llm=effective_use_llm, api_key=effective_key)
        builder = GraphBuilder(include_ontology_taxonomies=True)
        matcher = TopologicalMatcher()
        rag_engine = GraphRAGSummaryEngine(use_llm=effective_use_llm)

        # Ingest job description
        with st.spinner("Analyse sémantique de l'offre d'emploi..."):
            try:
                job_graph = extractor.extract_job(job_text, job_id="job_target", title="Poste Cible")
            except Exception as exc:
                st.warning(f"Avertissement lors de l'extraction de l'offre : {exc}")
                job_graph = extractor._extract_deterministic_job(job_text, "job_target", "Poste Cible")

        candidate_records: list[dict[str, Any]] = []
        parsing_errors: list[tuple[str, str]] = []

        progress = st.progress(0.0, text="Construction des graphes et audit des candidats en cours…")

        for i, cv_file in enumerate(cv_files):
            cand_name = Path(cv_file.name).stem.replace("_", " ").title()
            try:
                cv_text = extract_text(cv_file, filename=cv_file.name)

                # 1. Knowledge Graph Extraction
                try:
                    cand_graph = extractor.extract_candidate(cv_text, candidate_id=f"cand_{i+1}", candidate_name=cand_name)
                except Exception as e_ext:
                    st.warning(f"Mode dégradé d'extraction pour {cand_name}: {e_ext}")
                    cand_graph = extractor._extract_deterministic_candidate(cv_text, f"cand_{i+1}", cand_name)

                # 2. RDF & NetworkX Ingestion
                try:
                    rdf_g, nx_g = builder.build_candidate_graph(cand_graph)
                except Exception as e_bld:
                    st.warning(f"Erreur d'ingestion RDF pour {cand_name}: {e_bld}")
                    rdf_g = GraphBuilder(False).payload_to_rdflib(cand_graph)
                    nx_g = nx.MultiDiGraph()

                # 3. Contradiction & Inconsistency Audit
                try:
                    conflict_report = validate_candidate_graph(cand_graph)
                except Exception as e_val:
                    st.warning(f"Erreur d'audit pour {cand_name}: {e_val}")
                    conflict_report = None

                # 4. Topological Match
                try:
                    breakdown = matcher.compute_match(cand_graph, job_graph)
                except Exception as e_match:
                    st.warning(f"Erreur de matching topologique pour {cand_name}: {e_match}")
                    breakdown = None

                # 5. Hybrid Ranking using Unified HybridRanker
                ranker = HybridRanker(
                    weights=ranking_weights,
                    topological_matcher=matcher,
                    validator=validator,
                )
                ranking_result = ranker.score_candidate(cand_graph, job_graph)

                # 6. GraphRAG Critique
                try:
                    critique = rag_engine.generate_candidate_critique(cand_graph, job_graph)
                except Exception:
                    critique = f"Audit synthétique indisponible pour {cand_name}."

                candidate_records.append({
                    "name": cand_name,
                    "file_name": cv_file.name,
                    "cv_text": cv_text,
                    "cand_graph": cand_graph,
                    "rdf_graph": rdf_g,
                    "nx_graph": nx_g,
                    "conflict_report": conflict_report,
                    "graph_breakdown": ranking_result.graph_breakdown,
                    "vector_similarity": ranking_result.vector_similarity,
                    "experience_score": ranking_result.experience_score,
                    "degree_score": ranking_result.degree_score,
                    "missing_must_requirements": ranking_result.missing_must_requirements,
                    "is_capped_by_must_have": ranking_result.is_capped_by_must_have,
                    "final_score": ranking_result.final_score,
                    "critique": critique,
                })

            except DocumentParsingError as exc:
                parsing_errors.append((cv_file.name, str(exc)))
            except Exception as general_exc:
                parsing_errors.append((cv_file.name, f"Erreur inattendue: {general_exc}"))

            progress.progress((i + 1) / len(cv_files), text=f"Analyse en cours… ({i+1}/{len(cv_files)})")

        progress.empty()

        if parsing_errors:
            with st.expander(f"⚠️ {len(parsing_errors)} fichier(s) ont rencontré des erreurs", expanded=True):
                for name, err in parsing_errors:
                    st.warning(f"**{name}** — {err}")

        if not candidate_records:
            st.error("Aucun candidat n'a pu être traité.")
            st.stop()

        candidate_records.sort(key=lambda c: c["final_score"], reverse=True)

        st.subheader("📋 Résumé de l'offre & du vivier")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Compétences Offre", len(job_graph.get_skills()))
        c2.metric("Candidats analysés", len(candidate_records))
        c3.metric("Meilleur Score", f"{candidate_records[0]['final_score']}/100")
        avg_score = sum(c["final_score"] for c in candidate_records) / len(candidate_records)
        c4.metric("Moyenne du Vivier", f"{avg_score:.1f}/100")

        st.divider()

        st.subheader("🏆 Tableau Comparatif des Candidats")
        table_rows = []
        for rank, cand in enumerate(candidate_records, start=1):
            cr = cand["conflict_report"]
            bk = cand["graph_breakdown"]
            pen_display = f"-{cr.penalty_score:.2f}" if cr and cr.penalty_score > 0 else "0.00"
            status_audit = "✅ Valide" if (cr and cr.is_valid and not cr.conflicts) else f"⚠️ {len(cr.conflicts)} anomalie(s)" if cr else "Inconnu"

            missing_disp = ", ".join(cand["missing_must_requirements"]) if cand.get("missing_must_requirements") else "Aucune"
            table_rows.append({
                "Rang": rank,
                "Candidat": cand["name"],
                "Score Final": cand["final_score"],
                "Match Graphe (PPR+SKOS)": f"{bk.graph_match_score:.2f}" if bk else "N/A",
                "Similarité Vecteur": f"{cand['vector_similarity']:.2f}",
                "Expérience (Durée)": f"{cand['experience_score']:.2f}",
                "Diplôme (Niveau)": f"{cand['degree_score']:.2f}",
                "Pénalité Conflits": pen_display,
                "Statut Audit": status_audit,
                "Exigences Manquantes": missing_disp,
            })

        df_rank = pd.DataFrame(table_rows)
        st.dataframe(
            df_rank,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score Final": st.column_config.ProgressColumn("Score Final", min_value=0, max_value=100, format="%.1f"),
            },
        )

        csv_data = df_rank.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter le classement (CSV)", csv_data, "classement_knowledge_graph.csv", "text/csv")

        st.divider()

        st.subheader("🔬 Inspection Détaillée par Candidat")
        selected_cand_name = st.selectbox(
            "Sélectionnez un candidat à inspecter en détail :",
            [c["name"] for c in candidate_records],
            index=0,
        )

        cand = next(c for c in candidate_records if c["name"] == selected_cand_name)
        cr = cand["conflict_report"]
        bk = cand["graph_breakdown"]
        rdf_g = cand["rdf_graph"]
        nx_g = cand["nx_graph"]

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Final Ranking",
            "🛡️ Audit & Inconsistencies",
            "🔍 SPARQL Inspector",
            "🕸️ Graph Topology",
        ])

        with tab1:
            st.markdown(f"### Évaluation Globale : {cand['name']}")
            if cand.get("is_capped_by_must_have"):
                missing_str = ", ".join(cand.get("missing_must_requirements", []))
                st.warning(f"⚠️ **Score plafonné à 60.0%** : Compétence(s) obligatoire(s) manquante(s) : **{missing_str}**")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Final Score", f"{cand['final_score']} / 100")
            m2.metric("Vecteur (TF-IDF)", f"{cand['vector_similarity']:.2f}")
            m3.metric("Match Graphe", f"{bk.graph_match_score:.2f}" if bk else "0.00")
            m4.metric("Expérience", f"{cand['experience_score']:.2f}")
            m5.metric("Diplôme", f"{cand['degree_score']:.2f}")
            pen_val = f"-{cr.penalty_score:.2f}" if cr and cr.penalty_score > 0 else "0.00"
            m6.metric("Pénalité", pen_val)

            st.write("")
            st.markdown("#### 📑 Synthèse Explicative & Trajectoire (GraphRAG)")
            st.markdown(cand["critique"])

            if bk:
                st.write("")
                st.markdown("#### 🎯 Détail des Compétences Appariées")
                col_d, col_i = st.columns(2)
                with col_d:
                    st.markdown("**Correspondances Directes :**")
                    if bk.matched_skills_exact:
                        for s in bk.matched_skills_exact:
                            st.markdown(f"- ✅ **{s}**")
                    else:
                        st.caption("Aucune correspondance exacte.")

                with col_i:
                    st.markdown("**Compétences Déduites (Ontologie SKOS) :**")
                    if bk.matched_skills_inferred:
                        for inf in bk.matched_skills_inferred:
                            st.markdown(
                                f"- 🔄 `{inf['candidate_skill']}` ➔ `{inf['jd_skill']}` "
                                f"(Distance : {inf['distance']} sauts, Sim : {inf['similarity']:.2f})"
                            )
                    else:
                        st.caption("Aucune compétence déduite.")

        with tab2:
            st.markdown(f"### Audit d'Intégrité & Détection des Contradictions : {cand['name']}")
            if not cr or not cr.conflicts:
                st.markdown(
                    '<div class="clean-card"><strong>✅ Profil Intègre & Cohérent</strong><br>'
                    'Aucune inversion temporelle, anomalie chronologique ou sur-déclaration de compétence détectée.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="conflict-card"><strong>⚠️ {len(cr.conflicts)} Incohérence(s) Détectée(s) '
                    f'(Facteur de pénalité appliqué : -{cr.penalty_score:.2f})</strong></div>',
                    unsafe_allow_html=True,
                )
                for idx, conflict in enumerate(cr.conflicts, start=1):
                    ctype = conflict.get("conflict_type", "ANOMALY")
                    csev = conflict.get("severity", "MEDIUM")
                    msg = conflict.get("message", "")
                    details = conflict.get("details", {})
                    with st.expander(f"⚠️ Alerte #{idx} : [{ctype}] {csev}", expanded=True):
                        st.markdown(f"**Description :** {msg}")
                        if details:
                            st.json(details)

        with tab3:
            st.markdown(f"### Inspecteur de Requêtes SPARQL : Graphe de {cand['name']}")
            query_choice = st.selectbox(
                "Choisissez un type de requête SPARQL :",
                [
                    "1. Contrôle de confiance d'extraction (confidence < seuil)",
                    "2. Compétences liées à un rôle spécifique",
                    "3. Expansion taxonomique transitive (skos:broader*)",
                    "4. Éditeur SPARQL libre",
                ],
            )

            if query_choice.startswith("1."):
                threshold = st.slider("Seuil d'alerte de confiance (confidence < seuil)", 0.0, 1.0, 0.70, step=0.05)
                low_conf = get_low_confidence_relations(rdf_g, threshold=threshold)
                st.markdown(f"**Résultats ({len(low_conf)} relation(s) isolée(s)) :**")
                if low_conf:
                    st.dataframe(pd.DataFrame(low_conf), use_container_width=True)
                else:
                    st.success("Toutes les relations ont une confiance supérieure ou égale au seuil.")

            elif query_choice.startswith("2."):
                role_kw = st.text_input("Mot-clé ou intitulé de poste à filtrer (Regex) :", "Engineer")
                role_skills = get_skills_for_role(rdf_g, role_regex=role_kw)
                st.markdown(f"**Résultats ({len(role_skills)} compétence(s) trouvée(s)) :**")
                if role_skills:
                    st.dataframe(pd.DataFrame(role_skills), use_container_width=True)
                else:
                    st.info(f"Aucune compétence trouvée pour l'intitulé '{role_kw}'.")

            elif query_choice.startswith("3."):
                parent_concept = st.selectbox(
                    "Catégorie parente cible pour l'expansion transitive :",
                    ["machinelearning", "deeplearning", "backenddevelopment", "frontenddevelopment", "clouddevops", "data_infrastructure"],
                )
                tax_matches = match_skills_with_taxonomy(rdf_g, target_parent_skill=parent_concept)
                st.markdown(f"**Résultats ({len(tax_matches)} compétence(s) affiliée(s)) :**")
                if tax_matches:
                    st.dataframe(pd.DataFrame(tax_matches), use_container_width=True)
                else:
                    st.info(f"Aucune compétence ne remonte vers le concept parent '{parent_concept}'.")

            else:
                default_custom_q = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cv: <http://recruitment.org/cv#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 25"""
                custom_q = st.text_area("Saisissez votre requête SPARQL :", default_custom_q, height=130)
                if st.button("Exécuter la requête"):
                    try:
                        raw_res = rdf_g.query(custom_q)
                        rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in raw_res]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception as e_q:
                        st.error(f"Erreur d'exécution SPARQL : {e_q}")

        with tab4:
            st.markdown(f"### Topologie du Graphe de Connaissances : {cand['name']}")
            t1, t2, t3, t4 = st.columns(4)
            num_nodes = nx_g.number_of_nodes() if nx_g else 0
            num_edges = nx_g.number_of_edges() if nx_g else 0
            wcc = nx.number_weakly_connected_components(nx_g) if nx_g and num_nodes > 0 else 0
            density = nx.density(nx_g) if nx_g and num_nodes > 1 else 0.0

            t1.metric("Nœuds (Entités)", num_nodes)
            t2.metric("Arêtes (Relations)", num_edges)
            t3.metric("Composantes Connexes", wcc)
            t4.metric("Densité Topologique", f"{density:.3f}")

            st.write("")
            st.markdown("#### 🏷️ Répartition des Entités par Catégorie")
            cat_counts: dict[str, int] = {}
            for _, d in nx_g.nodes(data=True):
                c = d.get("category", "UNKNOWN")
                cat_counts[c] = cat_counts.get(c, 0) + 1

            if cat_counts:
                df_cats = pd.DataFrame([{"Catégorie": k, "Nombre": v} for k, v in cat_counts.items()])
                st.bar_chart(df_cats.set_index("Catégorie"))

            st.write("")
            st.markdown("#### 🔗 Liste des Relations Modélisées")
            edge_data = []
            for u, v, d in nx_g.edges(data=True):
                edge_data.append({
                    "Source": u,
                    "Prédicat": d.get("predicate_type", d.get("predicate", "")),
                    "Cible": v,
                    "Début": d.get("start_date") or "-",
                    "Fin": d.get("end_date") or "-",
                    "Confiance": f"{float(d.get('confidence', 1.0)):.2f}",
                })
            if edge_data:
                st.dataframe(pd.DataFrame(edge_data), use_container_width=True)

    else:
        st.info("Renseignez l'offre d'emploi et importez au moins un CV, puis lancez le classement et l'audit.")


# ==========================================================================
# MODE 2: TECHNICAL DOCUMENT & SPECIFICATION MODE
# ==========================================================================
else:
    st.title("📜 Technical Document & Specification Ingestion")
    st.caption(
        "Ingestion sémantique de standards, RFCs et spécifications techniques : extraction de dépendances "
        "de protocoles, modélisation de règles, et détection de contradictions de paramètres et contraintes circulaires."
    )

    st.subheader("1. Sélection d'une Spécification ou Chargement de Document")

    sample_dir = Path(__file__).parent / "data" / "sample_documents"
    presets_meta = {
        "RFC 7540 — HTTP/2 Framing, Multiplexing & Parameter Constraints": "rfc7540_http2_framing.txt",
        "OASIS MQTT 5.0 — IoT Protocol & Session Parameters": "mqtt_qos_specification.txt",
        "IEEE 802.11 — Wireless Cross-Layer Transport Excerpt": "ieee_wireless_paper_abstract.txt",
        "Saisie libre / Importer un document personnalisé": None,
    }

    selected_preset_label = st.selectbox(
        "Modèle de test rapide (1-Clic) :",
        list(presets_meta.keys()),
        index=0,
    )

    preset_filename = presets_meta[selected_preset_label]
    default_text = ""
    if preset_filename:
        file_path = sample_dir / preset_filename
        if file_path.exists():
            default_text = file_path.read_text(encoding="utf-8")

    col_mode1, col_mode2 = st.columns([1, 1])
    with col_mode1:
        doc_input_method = st.radio(
            "Mode d'entrée du document",
            ["Texte / Préréglage", "Importer un fichier (PDF, DOCX, TXT)"],
            horizontal=True,
        )

    doc_text: str = ""
    doc_title = selected_preset_label.split("—")[0].strip() if "—" in selected_preset_label else "Technical Specification"

    if doc_input_method == "Texte / Préréglage":
        doc_text = st.text_area(
            "Contenu de la spécification technique / RFC / Article de recherche :",
            value=default_text,
            height=220,
            placeholder="Collez ici le texte d'un RFC, standard ou document d'ingénierie...",
        )
    else:
        uploaded_doc = st.file_uploader(
            "Fichier de la spécification (PDF, DOCX ou TXT)",
            type=["pdf", "docx", "txt"],
            key="tech_doc_file",
        )
        if uploaded_doc is not None:
            try:
                doc_text = extract_text(uploaded_doc, filename=uploaded_doc.name)
                doc_title = Path(uploaded_doc.name).stem.replace("_", " ").title()
                with st.expander("Texte extrait du document"):
                    st.text(doc_text)
            except DocumentParsingError as exc:
                st.error(str(exc))

    run_doc = st.button("🔎 Analyser la Spécification & Construire le Graphe", type="primary")

    if run_doc:
        if not doc_text or not doc_text.strip():
            st.error("Merci de renseigner le document technique (texte ou fichier importé).")
            st.stop()

        extractor = KnowledgeGraphExtractor(use_llm=effective_use_llm, api_key=effective_key)
        builder = GraphBuilder(include_ontology_taxonomies=False)

        with st.spinner("Extraction des entités, concepts et relations de la spécification..."):
            try:
                doc_graph = extractor.extract_document(doc_text, document_id="doc_spec", document_title=doc_title)
            except Exception as e_ext:
                st.warning(f"Mode de repli déterministe activé : {e_ext}")
                doc_graph = extractor._extract_deterministic_document(doc_text, "doc_spec", doc_title)

        with st.spinner("Construction du graphe RDFLib et NetworkX MultiDiGraph..."):
            try:
                rdf_g, nx_g = builder.build_candidate_graph(doc_graph)
            except Exception as e_bld:
                st.warning(f"Erreur d'ingestion RDF : {e_bld}")
                rdf_g = builder.payload_to_rdflib(doc_graph)
                nx_g = nx.MultiDiGraph()

        with st.spinner("Audit de conformité et détection de contradictions de spécification..."):
            conflict_report = validate_document_graph(doc_graph)

        st.success(f"Analyse terminée avec succès : {len(doc_graph.entities)} entités et {len(doc_graph.relations)} relations extraites.")

        tab_d1, tab_d2, tab_d3, tab_d4 = st.tabs([
            "🕸️ Knowledge Graph & Dependencies",
            "🛡️ Specification Audit & Contradictions",
            "🔍 SPARQL Inspector",
            "📊 Document Overview & Synthesis",
        ])

        # TAB 1: KNOWLEDGE GRAPH & DEPENDENCIES
        with tab_d1:
            st.markdown(f"### Topologie de la Spécification : {doc_graph.name}")
            c1, c2, c3, c4 = st.columns(4)
            num_nodes = nx_g.number_of_nodes() if nx_g else 0
            num_edges = nx_g.number_of_edges() if nx_g else 0
            wcc = nx.number_weakly_connected_components(nx_g) if nx_g and num_nodes > 0 else 0
            density = nx.density(nx_g) if nx_g and num_nodes > 1 else 0.0

            c1.metric("Nœuds (Entités)", num_nodes)
            c2.metric("Arêtes (Relations)", num_edges)
            c3.metric("Composantes Connexes", wcc)
            c4.metric("Densité Topologique", f"{density:.3f}")

            st.write("")
            st.markdown("#### 🏷️ Répartition des Entités Techniques par Catégorie")
            cat_counts = {}
            for _, d in nx_g.nodes(data=True):
                cat = d.get("category", "CONCEPT")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            if cat_counts:
                df_cats = pd.DataFrame([{"Catégorie": k, "Nombre": v} for k, v in cat_counts.items()])
                st.bar_chart(df_cats.set_index("Catégorie"))

            st.write("")
            st.markdown("#### 🔗 Relations et Dépendances Modélisées")
            edge_rows = []
            for u, v, d in nx_g.edges(data=True):
                pred = d.get("predicate_type", d.get("predicate", ""))
                val = d.get("metadata", {}).get("value") if isinstance(d.get("metadata"), dict) else d.get("value")
                edge_rows.append({
                    "Entité Source": u,
                    "Prédicat": pred,
                    "Entité Cible": v,
                    "Valeur / Paramètre": val or "-",
                    "Snippet Source": d.get("source_snippet") or "-",
                    "Confiance": f"{float(d.get('confidence', 1.0)):.2f}",
                })
            if edge_rows:
                st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)

        # TAB 2: SPECIFICATION AUDIT & CONTRADICTIONS
        with tab_d2:
            st.markdown("### Audit de Cohérence & Détection des Contradictions")
            if not conflict_report.conflicts:
                st.markdown(
                    '<div class="clean-card"><strong>✅ Spécification Cohérente & Intègre</strong><br>'
                    'Aucun conflit de protocole, contrainte circulaire ou incohérence de paramètre détecté.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="conflict-card"><strong>⚠️ {len(conflict_report.conflicts)} Anomalie(s) de Spécification '
                    f'Détectée(s) (Facteur de pénalité : -{conflict_report.penalty_score:.2f})</strong></div>',
                    unsafe_allow_html=True,
                )
                for idx, conflict in enumerate(conflict_report.conflicts, start=1):
                    ctype = conflict.get("conflict_type", "ANOMALY")
                    csev = conflict.get("severity", "MEDIUM")
                    msg = conflict.get("message", "")
                    details = conflict.get("details", {})
                    with st.expander(f"⚠️ Alerte #{idx} : [{ctype}] {csev}", expanded=True):
                        st.markdown(f"**Description :** {msg}")
                        if details:
                            st.json(details)

        # TAB 3: SPARQL INSPECTOR
        with tab_d3:
            st.markdown("### Inspecteur SPARQL pour Spécifications Techniques")
            doc_q_choice = st.selectbox(
                "Sélectionnez une requête SPARQL cible :",
                [
                    "1. Dépendances de protocoles & concepts (cv:dependsOn)",
                    "2. Incompatibilités & conflits de protocoles (cv:conflictsWith)",
                    "3. Paramètres définis par section (cv:defines)",
                    "4. Détection de contraintes circulaires (cv:dependsOn+)",
                    "5. Éditeur SPARQL libre",
                ],
            )

            if doc_q_choice.startswith("1."):
                st.code(
                    """PREFIX cv: <http://recruitment.org/cv#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?concept ?conceptLabel ?specification ?specificationLabel WHERE {
    ?concept cv:dependsOn ?specification .
    OPTIONAL { ?concept rdfs:label ?conceptLabel }
    OPTIONAL { ?specification rdfs:label ?specificationLabel }
}""",
                    language="sparql",
                )
                deps = get_document_dependencies(rdf_g)
                st.markdown(f"**Résultats ({len(deps)} dépendance(s) identifiée(s)) :**")
                if deps:
                    st.dataframe(pd.DataFrame(deps), use_container_width=True)
                else:
                    st.info("Aucune dépendance explicite trouvée.")

            elif doc_q_choice.startswith("2."):
                st.code(
                    """PREFIX cv: <http://recruitment.org/cv#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?p1 ?p1Label ?p2 ?p2Label WHERE {
    ?p1 cv:conflictsWith ?p2 .
    OPTIONAL { ?p1 rdfs:label ?p1Label }
    OPTIONAL { ?p2 rdfs:label ?p2Label }
}""",
                    language="sparql",
                )
                confs = get_protocol_conflicts(rdf_g)
                st.markdown(f"**Résultats ({len(confs)} conflit(s) de protocoles isolé(s)) :**")
                if confs:
                    st.dataframe(pd.DataFrame(confs), use_container_width=True)
                else:
                    st.success("Aucune incompatibilité de protocole trouvée.")

            elif doc_q_choice.startswith("3."):
                st.code(
                    """PREFIX cv: <http://recruitment.org/cv#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?section ?sectionLabel ?parameter ?parameterLabel WHERE {
    ?section cv:defines ?parameter .
    OPTIONAL { ?section rdfs:label ?sectionLabel }
    OPTIONAL { ?parameter rdfs:label ?parameterLabel }
}""",
                    language="sparql",
                )
                params = get_defined_parameters(rdf_g)
                st.markdown(f"**Résultats ({len(params)} paramètre(s) défini(s)) :**")
                if params:
                    st.dataframe(pd.DataFrame(params), use_container_width=True)
                else:
                    st.info("Aucun paramètre défini trouvé.")

            elif doc_q_choice.startswith("4."):
                st.code(
                    """PREFIX cv: <http://recruitment.org/cv#>

SELECT DISTINCT ?a ?b WHERE {
    ?a cv:dependsOn+ ?b .
    ?b cv:dependsOn+ ?a .
    FILTER(?a != ?b)
}""",
                    language="sparql",
                )
                circs = find_circular_constraints(rdf_g)
                st.markdown(f"**Résultats ({len(circs)} contrainte(s) circulaire(s)) :**")
                if circs:
                    st.dataframe(pd.DataFrame(circs), use_container_width=True)
                else:
                    st.success("Aucune boucle de dépendance circulaire détectée.")

            else:
                default_doc_q = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cv: <http://recruitment.org/cv#>

SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 25"""
                custom_doc_q = st.text_area("Saisissez votre requête SPARQL :", default_doc_q, height=120)
                if st.button("Exécuter SPARQL"):
                    try:
                        raw_res = rdf_g.query(custom_doc_q)
                        rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in raw_res]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception as e_q:
                        st.error(f"Erreur d'exécution SPARQL : {e_q}")

        # TAB 4: DOCUMENT OVERVIEW & SYNTHESIS
        with tab_d4:
            st.markdown(f"### Synthèse Documentaire : {doc_graph.name}")
            st.write(f"**Résumé d'intégrité :** {conflict_report.summary}")

            st.write("")
            st.markdown("#### ⚙️ Dictionnaire des Paramètres de Configuration")
            param_list = []
            for ent in doc_graph.entities:
                if ent.category.value == "PARAMETER":
                    param_list.append({
                        "Paramètre": ent.name,
                        "Identifiant": ent.id,
                        "Valeur Définie": ent.metadata.get("value", "Non précisée"),
                    })
            if param_list:
                st.dataframe(pd.DataFrame(param_list), use_container_width=True)
            else:
                st.caption("Aucun paramètre explicite recensé.")

            st.write("")
            st.markdown("#### 🧱 Pile Protocolaire & Concepts Détectés")
            col_p, col_c = st.columns(2)
            with col_p:
                st.markdown("**Standards & Protocoles :**")
                spec_ents = [e for e in doc_graph.entities if e.category.value in {"SPECIFICATION", "PROTOCOL"}]
                if spec_ents:
                    for s in spec_ents:
                        st.markdown(f"- 📦 **{s.name}** (`{s.category.value}`)")
                else:
                    st.caption("Aucun protocole identifié.")

            with col_c:
                st.markdown("**Concepts Architecturaux :**")
                con_ents = [e for e in doc_graph.entities if e.category.value == "CONCEPT"]
                if con_ents:
                    for c in con_ents:
                        st.markdown(f"- 💡 **{c.name}**")
                else:
                    st.caption("Aucun concept architectural identifié.")

    else:
        st.info("Sélectionnez un modèle de test rapide ou importez une spécification, puis cliquez sur 'Analyser la Spécification'.")