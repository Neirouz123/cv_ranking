"""
app.py
------
Streamlit interface for the CV Ranker: upload a job offer and one or more
résumés (PDF/DOCX/TXT), and get an objective, weighted ranking of candidates
with a transparent breakdown of strengths and weaknesses for each one.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.extraction.extractor import ExtractedProfile
from src.parsing.document_parser import DocumentParsingError, extract_text
from src.scoring.scorer import ScoreBreakdown, ScoringWeights
from src.graph.pipeline import graph

st.set_page_config(
    page_title="Dossier RH — Classement de candidats",
    page_icon="🗂️",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling — lightweight "case file" theme consistent with the rest of the tool
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗂️ Dossier RH — Classement de candidats")
st.caption(
    "Déposez une offre d'emploi et un ou plusieurs CV (PDF, DOCX ou texte). "
    "Chaque candidat est noté sur des critères transparents et pondérables : "
    "compétences, pertinence textuelle, expérience et formation."
)

# --------------------------------------------------------------------------
# Sidebar — scoring weights
# --------------------------------------------------------------------------
st.sidebar.header("⚖️ Pondération du score")
st.sidebar.caption("Ajustez l'importance relative de chaque critère (normalisée automatiquement sur 100 %).")

w_skills = st.sidebar.slider("Compétences", 0, 100, 50, help="Correspondance des compétences détectées avec celles requises par l'offre.")
w_text = st.sidebar.slider("Pertinence textuelle", 0, 100, 20, help="Similarité générale du contenu du CV avec l'offre (TF-IDF).")
w_experience = st.sidebar.slider("Expérience", 0, 100, 15, help="Années d'expérience du candidat vs. années requises.")
w_education = st.sidebar.slider("Formation", 0, 100, 15, help="Niveau d'études du candidat vs. niveau requis.")

total_weight = w_skills + w_text + w_experience + w_education
if total_weight == 0:
    st.sidebar.error("Au moins un critère doit avoir un poids supérieur à 0.")
    st.stop()

st.sidebar.caption(
    f"Poids normalisés : compétences {w_skills/total_weight:.0%} · "
    f"texte {w_text/total_weight:.0%} · expérience {w_experience/total_weight:.0%} · "
    f"formation {w_education/total_weight:.0%}"
)

weights = ScoringWeights(
    skills=w_skills,
    text_relevance=w_text,
    experience=w_experience,
    education=w_education,
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Comment lire les scores ?**\n\n"
    "- 🟢 ≥ 75 : forte correspondance\n"
    "- 🟠 50–74 : correspondance partielle\n"
    "- 🔴 < 50 : correspondance faible"
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
        height=220,
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

run = st.button("🔎 Lancer le classement", type="primary", use_container_width=False)

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

    # The job offer only needs the lightweight regex/TF-IDF extraction for
    # this summary panel — no LLM call needed here, so it's done directly
    # rather than through the graph (which recomputes it per candidate as
    # part of each invoke, in parallel with extract_cv).
    job_profile = ExtractedProfile.from_text(job_text)

    results: list[tuple[str, ScoreBreakdown, str]] = []
    parsing_errors: list[tuple[str, str]] = []

    progress = st.progress(0.0, text="Analyse des CV en cours…")
    for i, cv_file in enumerate(cv_files):
        try:
            cv_text = extract_text(cv_file, filename=cv_file.name)
            graph_result = graph.invoke({"job_text": job_text, "cv_text": cv_text, "weights": weights})
            breakdown: ScoreBreakdown = graph_result["score_breakdown"]
            explanation: str = graph_result.get("explanation", "")
            results.append((cv_file.name, breakdown, explanation))
        except DocumentParsingError as exc:
            parsing_errors.append((cv_file.name, str(exc)))
        progress.progress((i + 1) / len(cv_files), text=f"Analyse en cours… ({i+1}/{len(cv_files)})")
    progress.empty()

    if parsing_errors:
        with st.expander(f"⚠️ {len(parsing_errors)} fichier(s) n'ont pas pu être analysés", expanded=True):
            for name, err in parsing_errors:
                st.warning(f"**{name}** — {err}")

    if not results:
        st.error("Aucun CV n'a pu être analysé.")
        st.stop()

    results.sort(key=lambda r: r[1].overall_score, reverse=True)

    # ---- Job offer summary ----
    st.subheader("📋 Résumé de l'offre analysée")
    c1, c2, c3 = st.columns(3)
    c1.metric("Compétences identifiées", len(job_profile.skills_flat))
    c2.metric("Expérience requise", f"{job_profile.experience_years} an(s)" if job_profile.experience_years else "Non précisée")
    c3.metric("Formation requise", job_profile.education_level[1] if job_profile.education_level else "Non précisée")
    if job_profile.skills_flat:
        st.caption("Compétences détectées dans l'offre : " + ", ".join(sorted(job_profile.skills_flat)))

    st.divider()

    # ---- Ranking table ----
    st.subheader("🏆 Classement des candidats")
    table_rows = []
    for rank, (name, r, _explanation) in enumerate(results, start=1):
        table_rows.append(
            {
                "Rang": rank,
                "Candidat": name,
                "Score global": r.overall_score,
                "Compétences": r.skills_score,
                "Texte": r.text_relevance_score,
                "Expérience": r.experience_score,
                "Formation": r.education_score,
                "Compétences manquantes": sum(len(v) for v in r.missing_skills.values()),
            }
        )
    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score global": st.column_config.ProgressColumn(
                "Score global", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger le classement (CSV)", csv, "classement_candidats.csv", "text/csv")

    st.divider()

    # ---- Per-candidate detail ----
    st.subheader("🗂️ Détail par candidat")
    for rank, (name, r, explanation) in enumerate(results, start=1):
        badge_class = "rank-1" if r.overall_score >= 75 else ("rank-2" if r.overall_score >= 50 else "rank-3")
        with st.expander(f"#{rank} — {name} · score {r.overall_score}/100", expanded=(rank == 1)):
            st.markdown(
                f'<span class="score-badge {badge_class}">Score global : {r.overall_score}/100</span>',
                unsafe_allow_html=True,
            )
            st.write("")

            if explanation:
                st.markdown("**🧾 Synthèse**")
                st.markdown(explanation)
                st.write("")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Compétences", f"{r.skills_score}/100")
            sc2.metric("Pertinence texte", f"{r.text_relevance_score}/100")
            sc3.metric("Expérience", f"{r.experience_score}/100")
            sc4.metric("Formation", f"{r.education_score}/100")

            col_strengths, col_weaknesses = st.columns(2)
            with col_strengths:
                st.markdown("**✅ Points forts**")
                if r.strengths:
                    for s in r.strengths:
                        st.markdown(f"- {s}")
                else:
                    st.caption("Aucun point fort notable identifié.")
            with col_weaknesses:
                st.markdown("**⚠️ Points faibles**")
                if r.weaknesses:
                    for w in r.weaknesses:
                        st.markdown(f"- {w}")
                else:
                    st.caption("Aucun point faible notable identifié.")

            if r.matched_skills:
                st.markdown("**Compétences correspondantes, par catégorie :**")
                for cat, skills in r.matched_skills.items():
                    st.markdown(f"- *{cat}* : {', '.join(skills)}")

            if r.missing_skills:
                st.markdown("**Compétences manquantes, par catégorie :**")
                for cat, skills in r.missing_skills.items():
                    st.markdown(f"- *{cat}* : {', '.join(skills)}")

else:
    st.info("Renseignez l'offre d'emploi et importez au moins un CV, puis lancez le classement.")