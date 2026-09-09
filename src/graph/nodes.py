"""
nodes.py
--------
Step 3: llm_extract_cv now makes a real call to Groq (Llama 3.3 70B) for
structured extraction, replacing the Step 2 stub. Falls back to the
regex-based profile if the API call or JSON parsing fails, so the app
never hard-crashes on an LLM hiccup.
"""

import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.extraction.extractor import ExtractedProfile, find_skills, flatten_skills
from src.extraction.skills_data import all_skills_flat, skill_category
from src.scoring.scorer import score_candidate
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