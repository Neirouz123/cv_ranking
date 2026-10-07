"""
--------
nodes.py
--------
"""

import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.extraction.extractor import ExtractedProfile, find_skills, flatten_skills
from src.extraction.skills_data import all_skills_flat, skill_category
from src.scoring.scorer import ScoreBreakdown, score_candidate
from src.graph.state import RankingState

load_dotenv()

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)

_EXTRACTION_PROMPT = """Tu es un assistant d'extraction de données RH. Analyse le texte de CV suivant et retourne UNIQUEMENT un objet JSON valide, sans aucun texte avant ou après, avec exactement cette structure :

{{
  "skills": ["compétence1", "compétence2", ...],
  "experience_years": <nombre entier ou null>,
  "education_level": <nombre entier de 1 à 5 ou null>
}}

Où education_level correspond à :
1 = Baccalauréat, 2 = Bac+2 (BTS/DUT), 3 = Licence/Bachelor, 4 = Master/Bac+5, 5 = Doctorat/PhD

Instructions :
- "skills" : liste les compétences techniques et outils mentionnés (langages, frameworks, méthodologies).
- "experience_years" : le nombre total d'années d'expérience professionnelle, déduit du texte même si non explicitement énoncé (ex: à partir des dates ou du parcours). Mets null si impossible à déterminer.
- "education_level" : le niveau de diplôme le plus élevé mentionné. Mets null si aucun diplôme n'est mentionné.

Texte du CV :
---
{cv_text}
---

Réponds uniquement avec le JSON, rien d'autre."""


def extract_job(state: RankingState) -> dict:
    profile = ExtractedProfile.from_text(state["job_text"])
    return {"job_profile": profile}


def extract_cv(state: RankingState) -> dict:
    profile = ExtractedProfile.from_text(state["cv_text"])
    return {"cv_profile": profile}


def check_extraction_confidence(state: RankingState) -> str:
    profile = state["cv_profile"]
    if not profile.skills_flat or profile.experience_years is None or profile.education_level is None:
        return "llm_extract_cv"
    return "compute_score"


def _map_llm_skills_to_taxonomy(llm_skills: list[str]) -> dict[str, list[str]]:
    """
    The LLM returns free-text skill names. We re-run them through the
    existing regex-based skill detector (on the LLM's own output, treated
    as a mini-document) so results stay grouped by the same taxonomy
    categories the scorer expects — keeps skills_by_category consistent
    whether extraction came from regex or LLM.
    """
    joined_text = ", ".join(llm_skills)
    return find_skills(joined_text)


_EDUCATION_LABELS = {
    1: "Baccalauréat",
    2: "Bac+2 (BTS/DUT)",
    3: "Licence / Bachelor",
    4: "Master / Bac+5",
    5: "Doctorat / PhD",
}


def llm_extract_cv(state: RankingState) -> dict:
    """
    Calls Groq (Llama 3.3 70B) to extract skills/experience/education from
    the CV text as structured JSON. Falls back to the existing regex-based
    profile (already computed by extract_cv) if the call or parsing fails.
    """
    fallback_profile = state["cv_profile"]
    cv_text = state["cv_text"]

    try:
        prompt = _EXTRACTION_PROMPT.format(cv_text=cv_text)
        response = _llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if the model added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            raw = raw.removeprefix("json").strip()

        data = json.loads(raw)

        skills_by_category = _map_llm_skills_to_taxonomy(data.get("skills", []))
        edu_level_int = data.get("education_level")
        education_level = (
            (edu_level_int, _EDUCATION_LABELS[edu_level_int])
            if edu_level_int in _EDUCATION_LABELS
            else None
        )

        profile = ExtractedProfile(
            raw_text=cv_text,
            skills_by_category=skills_by_category,
            skills_flat=flatten_skills(skills_by_category),
            experience_years=data.get("experience_years"),
            education_level=education_level,
        )
        return {"cv_profile": profile}

    except Exception as exc:
        print(f"⚠️  llm_extract_cv failed ({exc}), falling back to regex profile.")
        return {"cv_profile": fallback_profile}


def compute_score(state: RankingState) -> dict:
    breakdown = score_candidate(
        cv_profile=state["cv_profile"],
        job_profile=state["job_profile"],
        weights=state.get("weights"),
    )
    return {"score_breakdown": breakdown}


# --------------------------------------------------------------------------
# Step 4: natural-language explanation
# --------------------------------------------------------------------------

_EXPLANATION_PROMPT = """Tu es un(e) assistant(e) RH qui rédige des synthèses de candidature claires et professionnelles à destination de recruteurs.

Voici l'analyse d'un candidat pour un poste :

- Correspondance des compétences : {skills_qualifier} ({matched_count} compétence(s) correspondante(s) sur {total_count} requise(s))
- Pertinence générale du profil par rapport à l'offre : {text_qualifier}
- Expérience : candidat {candidate_exp}, poste requiert {required_exp} ({experience_qualifier})
- Formation : candidat {candidate_edu}, poste requiert {required_edu} ({education_qualifier})

Points forts identifiés :
{strengths}

Points faibles identifiés :
{weaknesses}

Rédige un paragraphe de synthèse de 4 à 7 phrases, en français, avec un ton professionnel et neutre, que pourrait lire un recruteur pressé pour se faire un premier avis sur ce candidat.

Consignes :
- Rédige des phrases complètes et fluides, pas de liste à puces.
- Ne mentionne aucun score chiffré brut (le recruteur les voit déjà ailleurs dans l'interface) : reformule-les qualitativement.
- Termine par une phrase de conclusion qui situe implicitement le candidat (à retenir en priorité / à considérer avec réserve / peu adapté au poste en l'état), sans jugement moral sur la personne.
- Réponds uniquement avec le paragraphe, sans titre, sans introduction, sans guillemets.
"""


def _qualifier(score: float) -> str:
    """Map a 0-100 subscore to a short qualitative French phrase for the prompt."""
    if score >= 85:
        return "excellente"
    if score >= 70:
        return "bonne"
    if score >= 50:
        return "partielle"
    if score >= 25:
        return "faible"
    return "très faible"


def _format_experience(years: int | None) -> str:
    return f"{years} an(s)" if years is not None else "non précisée"


def _format_education(level: tuple[int, str] | None) -> str:
    return level[1] if level is not None else "non précisée"


def _fallback_explanation(breakdown: ScoreBreakdown) -> str:
    """
    Deterministic templated paragraph built from the existing strengths /
    weaknesses bullets, used when the LLM call or its output is unusable.
    Not as fluent as the LLM version, but always available and still
    readable by a recruiter.
    """
    if breakdown.overall_score >= 75:
        opening = "Ce candidat présente une correspondance globale forte avec le poste."
    elif breakdown.overall_score >= 50:
        opening = "Ce candidat présente une correspondance globale partielle avec le poste."
    else:
        opening = "Ce candidat présente une correspondance globale faible avec le poste."

    sentences = [opening]
    if breakdown.strengths:
        sentences.append("Points positifs : " + " ".join(breakdown.strengths))
    if breakdown.weaknesses:
        sentences.append("Points à vérifier : " + " ".join(breakdown.weaknesses))

    return " ".join(sentences)


def generate_explanation(state: RankingState) -> dict:
    """
    Calls Groq (Llama 3.3 70B) to turn the structured ScoreBreakdown into a
    short, readable French paragraph. Falls back to a deterministic
    templated paragraph (from the strengths/weaknesses bullets already
    computed by compute_score) if the call or the response is unusable —
    same never-hard-crash pattern as llm_extract_cv.
    """
    breakdown: ScoreBreakdown = state["score_breakdown"]

    total_count = sum(len(v) for v in breakdown.matched_skills.values()) + sum(
        len(v) for v in breakdown.missing_skills.values()
    )
    matched_count = sum(len(v) for v in breakdown.matched_skills.values())

    strengths_block = "\n".join(f"- {s}" for s in breakdown.strengths) or "- Aucun point fort notable identifié."
    weaknesses_block = "\n".join(f"- {w}" for w in breakdown.weaknesses) or "- Aucun point faible notable identifié."

    prompt = _EXPLANATION_PROMPT.format(
        skills_qualifier=_qualifier(breakdown.skills_score),
        matched_count=matched_count,
        total_count=total_count,
        text_qualifier=_qualifier(breakdown.text_relevance_score),
        candidate_exp=_format_experience(breakdown.candidate_experience_years),
        required_exp=_format_experience(breakdown.required_experience_years),
        experience_qualifier=_qualifier(breakdown.experience_score),
        candidate_edu=_format_education(breakdown.candidate_education),
        required_edu=_format_education(breakdown.required_education),
        education_qualifier=_qualifier(breakdown.education_score),
        strengths=strengths_block,
        weaknesses=weaknesses_block,
    )

    try:
        response = _llm.invoke(prompt)
        explanation = response.content.strip().strip('"')
        if not explanation:
            raise ValueError("Réponse vide du modèle.")
        return {"explanation": explanation}
    except Exception as exc:
        print(f"⚠️  generate_explanation failed ({exc}), falling back to templated paragraph.")
        return {"explanation": _fallback_explanation(breakdown)}


# --------------------------------------------------------------------------
# Knowledge Graph & GraphRAG Pipeline Nodes
# --------------------------------------------------------------------------

def extract_node(state: RankingState) -> dict:
    """
    Extract structured knowledge graph entities and relations for candidate and job description.
    """
    from src.pipeline.extractor import KnowledgeGraphExtractor

    extractor = KnowledgeGraphExtractor(use_llm=False)
    cand_name = state.get("candidate_name") or "Candidate"

    extracted_cv = extractor.extract_candidate(state["cv_text"], candidate_id="cand_extracted", candidate_name=cand_name)
    extracted_job = extractor.extract_job(state["job_text"], job_id="job_extracted", title="Target Job")

    return {
        "extracted_cv": extracted_cv,
        "extracted_job": extracted_job,
    }


def graph_construction_node(state: RankingState) -> dict:
    """
    Ingest extracted knowledge into RDFLib and export to NetworkX MultiDiGraph with qualifiers.
    """
    from src.graph.builder import GraphBuilder

    builder = GraphBuilder(include_ontology_taxonomies=True)
    rdf_g, nx_g = builder.build_candidate_graph(state["extracted_cv"])

    return {
        "rdf_graph": rdf_g,
        "nx_graph": nx_g,
    }


def validation_node(state: RankingState) -> dict:
    """
    Run temporal inversion, anachronism, and cumulative over-claiming contradiction audits.
    """
    from src.core.validator import validate_candidate_graph

    conflict_report = validate_candidate_graph(state["extracted_cv"])
    return {
        "conflict_report": conflict_report,
    }


def ranking_node(state: RankingState) -> dict:
    """
    Compute Personalized PageRank, SKOS taxpath distance, hybrid score, and GraphRAG critique.
    """
    from src.graph.matching import TopologicalMatcher
    from src.graph.graph_rag import GraphRAGSummaryEngine
    from src.pipeline.ranker import HybridRanker

    matcher = TopologicalMatcher()
    breakdown = matcher.compute_match(state["extracted_cv"], state["extracted_job"])

    ranker = HybridRanker(topological_matcher=matcher)
    ranking_result = ranker.score_candidate(state["extracted_cv"], state["extracted_job"])

    rag_engine = GraphRAGSummaryEngine(use_llm=False)
    critique = rag_engine.generate_candidate_critique(state["extracted_cv"], state["extracted_job"])

    return {
        "graph_breakdown": breakdown,
        "ranking_result": ranking_result,
        "critique": critique,
    }