"""
app.py
------
Streamlit interface for the Knowledge Graph, SPARQL Audit & GraphRAG CV Ranker.
Upload a job offer and one or more résumés (PDF/DOCX/TXT) to obtain an objective,
hybrid ranking combining topological knowledge graphs, semantic text relevance,
and automated contradiction auditing.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.core.validator import validate_candidate_graph
from src.extraction.extractor import ExtractedProfile
from src.graph.builder import GraphBuilder
from src.graph.graph_rag import GraphRAGSummaryEngine
from src.graph.matching import TopologicalMatcher
from src.graph.pipeline import graph, kg_pipeline
from src.graph.sparql_queries import (
    get_low_confidence_relations,
    get_skills_for_role,
    match_skills_with_taxonomy,
)
from src.parsing.document_parser import DocumentParsingError, extract_text
from src.pipeline.extractor import KnowledgeGraphExtractor
from src.pipeline.ranker import HybridRanker, compute_final_rank, compute_vector_similarity
from src.scoring.scorer import ScoreBreakdown, ScoringWeights

st.set_page_config(
    page_title="Dossier RH — Knowledge Graph & GraphRAG Ranker",
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

st.title("🗂️ Dossier RH — Knowledge Graph & GraphRAG Ranker")
st.caption(
    "Système d'évaluation de candidats fondé sur un **graphe de connaissances ontologique (RDFLib / SKOS)**, "
    "du **scoring topologique (NetworkX / Personalized PageRank)**, un **audit d'incohérences temporelles & factuelles**, "
    "et des **synthèses qualitatives GraphRAG**."
)

# --------------------------------------------------------------------------
# Sidebar — scoring weights
# --------------------------------------------------------------------------
st.sidebar.header("⚖️ Pondération Hybride")
st.sidebar.caption("Ajustez l'importance des composantes du score hybride :")

w_vec_slider = st.sidebar.slider("Similarité Textuelle (Vecteur / TF-IDF)", 0, 100, 40, help="Pondération alpha.")
w_graph_slider = st.sidebar.slider("Match Topologique Graphe (PPR + SKOS)", 0, 100, 50, help="Pondération bêta.")
w_pen_slider = st.sidebar.slider("Pénalité d'Intégrité / Conflits", 0, 100, 10, help="Pondération gamma déductive.")

sum_pos = w_vec_slider + w_graph_slider
if sum_pos == 0:
    st.sidebar.error("Au moins un critère de matching doit être supérieur à 0.")
    st.stop()

w_vec = round(w_vec_slider / 100.0, 2)
w_graph = round(w_graph_slider / 100.0, 2)
w_pen = round(w_pen_slider / 100.0, 2)

st.sidebar.caption(
    f"Pondérations effectives : Vecteur {w_vec:.2f} · Graphe {w_graph:.2f} · Déduction conflit {w_pen:.2f}"
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Guide de lecture des scores :**\n\n"
    "- 🟢 **≥ 75** : Forte correspondance & intégrité validée\n"
    "- 🟠 **50–74** : Correspondance partielle / compétences connexes\n"
    "- 🔴 **< 50** : Profil éloigné ou pénalisé par des contradictions"
)

# --------------------------------------------------------------------------
# Job offer input
# --------------------------------------------------------------------------
st.subheader("1. Offre d'emploi")

job_input_mode = st.radio(
    "Méthode de saisie", ["Coller le texte", "Importer un fichier"], horizontal=True, key="job_mode"
)

job_text: str | None = None

if job_input_mode == "Coller le texte":
    job_text = st.text_area(
        "Texte de l'offre d'emploi",
        height=180,
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

# --------------------------------------------------------------------------
# Résumés input
# --------------------------------------------------------------------------
st.subheader("2. CV des candidats")
cv_files = st.file_uploader(
    "Importez un ou plusieurs CV (PDF, DOCX ou TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

run = st.button("🔎 Lancer le classement & l'audit", type="primary", use_container_width=False)

# --------------------------------------------------------------------------
# Processing & results
# --------------------------------------------------------------------------
if run:
    if not job_text or not job_text.strip():
        st.error("Merci de renseigner l'offre d'emploi (texte collé ou fichier importé).")
        st.stop()
    if not cv_files:
        st.error("Merci d'importer au moins un CV.")
        st.stop()

    extractor = KnowledgeGraphExtractor(use_llm=False)
    builder = GraphBuilder(include_ontology_taxonomies=True)
    matcher = TopologicalMatcher()
    rag_engine = GraphRAGSummaryEngine(use_llm=False)

    # Ingest job description
    with st.spinner("Analyse sémantique de l'offre d'emploi..."):
        try:
            job_graph = extractor.extract_job(job_text, job_id="job_target", title="Poste Cible")
            job_profile = ExtractedProfile.from_text(job_text)
        except Exception as exc:
            st.warning(f"Avertissement lors de l'extraction de l'offre : {exc}")
            job_graph = extractor.extract_job(job_text, job_id="job_target")
            job_profile = ExtractedProfile.from_text(job_text)

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

            # 5. Semantic Vector Match
            vec_sim = compute_vector_similarity(cv_text, job_text)

            # 6. Hybrid Score Computation
            graph_score = breakdown.graph_match_score if breakdown else 0.5
            pen_score = conflict_report.penalty_score if conflict_report else 0.0

            final_rank_0_1 = compute_final_rank(
                vector_score=vec_sim,
                graph_score=graph_score,
                penalty_score=pen_score,
                weights=(w_vec, w_graph, w_pen),
            )
            final_score_100 = round(final_rank_0_1 * 100.0, 1)

            # 7. GraphRAG Critique
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
                "graph_breakdown": breakdown,
                "vector_similarity": vec_sim,
                "final_score": final_score_100,
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

    # ---- Summary metric bar ----
    st.subheader("📋 Résumé de l'offre & du vivier")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compétences Offre", len(job_graph.get_skills()))
    c2.metric("Candidats analysés", len(candidate_records))
    c3.metric("Meilleur Score", f"{candidate_records[0]['final_score']}/100")
    avg_score = sum(c["final_score"] for c in candidate_records) / len(candidate_records)
    c4.metric("Moyenne du Vivier", f"{avg_score:.1f}/100")

    st.divider()

    # ---- Ranking overview table ----
    st.subheader("🏆 Tableau Comparatif des Candidats")
    table_rows = []
    for rank, cand in enumerate(candidate_records, start=1):
        cr = cand["conflict_report"]
        bk = cand["graph_breakdown"]
        pen_display = f"-{cr.penalty_score:.2f}" if cr and cr.penalty_score > 0 else "0.00"
        status_audit = "✅ Valide" if (cr and cr.is_valid and not cr.conflicts) else f"⚠️ {len(cr.conflicts)} anomalie(s)" if cr else "Inconnu"

        table_rows.append({
            "Rang": rank,
            "Candidat": cand["name"],
            "Score Final": cand["final_score"],
            "Match Graphe (PPR+SKOS)": f"{bk.graph_match_score:.2f}" if bk else "N/A",
            "Similarité Vecteur": f"{cand['vector_similarity']:.2f}",
            "Pénalité Conflits": pen_display,
            "Statut Audit": status_audit,
            "Compétences Directes": len(bk.matched_skills_exact) if bk else 0,
            "Compétences Déduites": len(bk.matched_skills_inferred) if bk else 0,
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

    # ---- Deep-Dive Tabbed Interface per Candidate ----
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

    # --------------------------------------------------------------------------
    # TAB 1: FINAL RANKING
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown(f"### Évaluation Globale : {cand['name']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Score", f"{cand['final_score']} / 100")
        m2.metric("Semantic Vector Match", f"{cand['vector_similarity']:.2f}")
        m3.metric("Graph / PPR Match", f"{bk.graph_match_score:.2f}" if bk else "0.00")
        pen_val = f"-{cr.penalty_score:.2f}" if cr and cr.penalty_score > 0 else "0.00"
        m4.metric("Audit Penalty", pen_val)

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

    # --------------------------------------------------------------------------
    # TAB 2: AUDIT & INCONSISTENCIES
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # TAB 3: SPARQL INSPECTOR
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown(f"### Inspecteur de Requêtes SPARQL : Graphe de {cand['name']}")
        st.caption("Exécutez des requêtes SPARQL interactives en temps réel sur le graphe RDFLib du candidat.")

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
            st.markdown("**Requête SPARQL exécutée :**")
            st.code(
                f"""PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cv: <http://recruitment.org/cv#>

SELECT ?stmt ?subject ?predicate ?object ?confidence ?sourceSnippet WHERE {{
    ?stmt rdf:type rdf:Statement ;
          rdf:subject ?subject ;
          rdf:predicate ?predicate ;
          rdf:object ?object ;
          cv:confidence ?confidence .
    OPTIONAL {{ ?stmt cv:sourceSnippet ?sourceSnippet }}
    FILTER(xsd:float(?confidence) < {threshold})
}}""",
                language="sparql",
            )
            low_conf = get_low_confidence_relations(rdf_g, threshold=threshold)
            st.markdown(f"**Résultats ({len(low_conf)} relation(s) isolée(s)) :**")
            if low_conf:
                st.dataframe(pd.DataFrame(low_conf), use_container_width=True)
            else:
                st.success("Toutes les relations ont une confiance supérieure ou égale au seuil.")

        elif query_choice.startswith("2."):
            role_kw = st.text_input("Mot-clé ou intitulé de poste à filtrer (Regex) :", "Engineer")
            st.markdown("**Requête SPARQL exécutée :**")
            st.code(
                f"""PREFIX cv: <http://recruitment.org/cv#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?role ?roleLabel ?skill ?skillLabel WHERE {{
    ?candidate (cv:hasExperience | cv:heldRole) ?role .
    ?role (cv:usesSkill | cv:usedSkill) ?skill .
    OPTIONAL {{ ?role rdfs:label ?roleLabel }}
    OPTIONAL {{ ?skill rdfs:label ?skillLabel }}
    FILTER(REGEX(STR(?roleLabel), "{role_kw}", "i"))
}}""",
                language="sparql",
            )
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
            st.markdown("**Requête SPARQL exécutée :**")
            st.code(
                f"""PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX cv: <http://recruitment.org/cv#>

SELECT DISTINCT ?skill ?skillLabel ?parent WHERE {{
    ?candidate (cv:usesSkill | cv:hasExperience/cv:usesSkill) ?skill .
    ?skill skos:broader* ?parent .
    OPTIONAL {{ ?skill rdfs:label ?skillLabel }}
    FILTER(REGEX(STR(?parent), "{parent_concept}", "i"))
}}""",
                language="sparql",
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
            custom_q = st.text_area("Saisissez votre requête SPARQL :", default_custom_q, height=140)
            if st.button("Exécuter la requête"):
                try:
                    raw_res = rdf_g.query(custom_q)
                    rows = [
                        {str(k): str(v) for k, v in row.asdict().items()}
                        for row in raw_res
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception as e_q:
                    st.error(f"Erreur d'exécution SPARQL : {e_q}")

    # --------------------------------------------------------------------------
    # TAB 4: GRAPH TOPOLOGY
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown(f"### Topologie du Graphe de Connaissances : {cand['name']}")

        t1, t2, t3, t4 = st.columns(4)
        num_nodes = nx_g.number_of_nodes() if nx_g else 0
        num_edges = nx_g.number_of_edges() if nx_g else 0

        # Weakly connected components
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
        else:
            st.caption("Aucun nœud isolé.")

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