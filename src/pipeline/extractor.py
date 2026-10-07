"""
src/pipeline/extractor.py
-------------------------
LLM-based relation & temporal extraction pipeline for CVs and Job Descriptions.
Extracts structured knowledge triplets (Subject-Predicate-Object with temporal qualifiers)
into strict Pydantic schemas. Includes deterministic offline fallback with fuzzy date parsing.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Optional

from dotenv import load_dotenv

from src.core.models import (
    CandidateGraphPayload,
    Entity,
    EntityCategory,
    ExtractedGraph,
    JobDescriptionGraphPayload,
    PredicateType,
    Relation,
)
from src.core.ontology import normalize_skill_name
from src.core.validator import parse_fuzzy_date

load_dotenv()

# Structured extraction prompt for LLM tool calling or JSON mode
RELATION_EXTRACTION_PROMPT = """You are an expert Knowledge Graph Extraction system for recruitment and HR.
Analyze the following text and extract a structured knowledge graph in strictly valid JSON format.

CRITICAL INSTRUCTIONS:
1. Extract all key ENTITIES:
   - CANDIDATE: The person's name or id
   - ROLE: Job titles / positions held (e.g., "Senior Machine Learning Engineer")
   - COMPANY: Employing organizations
   - SKILL: Technical skills, frameworks, tools, libraries (e.g., "PyTorch", "FastAPI")
   - DEGREE: Academic qualifications (e.g., "Master of Science in Computer Science")
   - PROJECT: Specific initiatives or projects delivered
2. Extract RELATIONS with explicit temporal qualifiers, confidence, and source snippets:
   - Relate skills directly to the specific ROLE or PROJECT during which they were used (e.g. role -> USES_SKILL -> skill).
   - Relate candidate to roles (candidate -> HELD_ROLE -> role), companies (role -> WORKED_AT -> company), and degrees (candidate -> EARNED_DEGREE -> degree).
   - Extract exact start_date and end_date for each tenure/project (e.g., "2020-01", "2023-05", "2019", "present").
   - Extract claimed_years if explicitly mentioned (e.g., "5 years of Python").
   - Extract source_snippet containing the exact context from the text.
3. DO NOT hallucinate any entity or claim not explicitly supported by the text.

JSON Output Schema:
{{
  "candidate_name": "<full name>",
  "entities": [
    {{
      "id": "<category_prefix:name>",
      "name": "<display name>",
      "category": "CANDIDATE" | "ROLE" | "SKILL" | "DEGREE" | "COMPANY" | "PROJECT",
      "claimed_years": <number or null>
    }}
  ],
  "relations": [
    {{
      "subject_id": "<entity_id>",
      "predicate": "HELD_ROLE" | "WORKED_AT" | "USES_SKILL" | "DELIVERED_PROJECT" | "EARNED_DEGREE",
      "object_id": "<entity_id>",
      "start_date": "<YYYY-MM or YYYY or present or null>",
      "end_date": "<YYYY-MM or YYYY or present or null>",
      "confidence": <float between 0.0 and 1.0>,
      "source_snippet": "<verbatim excerpt from text>"
    }}
  ]
}}

Source Text:
---
{text}
---

Respond ONLY with the JSON object, with no markdown code fences or conversational text.
"""


class KnowledgeGraphExtractor:
    """
    Extracts structured entities, relations, and temporal qualifiers from CV and JD documents.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", use_llm: bool = True) -> None:
        self.model_name = model_name
        self.use_llm = use_llm
        self._llm = None

        api_key = os.environ.get("GROQ_API_KEY")
        if self.use_llm and api_key:
            try:
                from langchain_groq import ChatGroq
                self._llm = ChatGroq(model=self.model_name, api_key=api_key, temperature=0.0)
            except Exception:
                self._llm = None

    def extract_candidate(
        self,
        cv_text: str,
        candidate_id: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ) -> CandidateGraphPayload:
        """
        Extract knowledge graph representation for a candidate CV.
        """
        cid = candidate_id or f"cand_{uuid.uuid4().hex[:6]}"

        if self._llm is not None:
            try:
                return self._extract_with_llm_candidate(cv_text, cid, candidate_name)
            except Exception:
                pass  # Fall back smoothly to deterministic parser

        return self._extract_deterministic_candidate(cv_text, cid, candidate_name)

    def extract_job(
        self,
        job_text: str,
        job_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> JobDescriptionGraphPayload:
        """
        Extract knowledge graph representation for a job description.
        """
        jid = job_id or f"job_{uuid.uuid4().hex[:6]}"

        if self._llm is not None:
            try:
                return self._extract_with_llm_job(job_text, jid, title)
            except Exception:
                pass

        return self._extract_deterministic_job(job_text, jid, title)

    def _extract_with_llm_candidate(
        self, cv_text: str, candidate_id: str, candidate_name: Optional[str]
    ) -> CandidateGraphPayload:
        prompt = RELATION_EXTRACTION_PROMPT.format(text=cv_text)
        res = self._llm.invoke(prompt)
        content = res.content.strip()

        # Clean JSON markdown if wrapped
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)
        name = candidate_name or data.get("candidate_name") or candidate_id

        entities: list[Entity] = []
        for e in data.get("entities", []):
            cat_str = str(e.get("category", "SKILL")).upper()
            if cat_str in EntityCategory.__members__:
                cat = EntityCategory[cat_str]
            else:
                cat = EntityCategory.SKILL
            entities.append(
                Entity(
                    id=e["id"],
                    label=e.get("label", e["id"]),
                    category=cat,
                    claimed_years=e.get("claimed_years"),
                )
            )

        relations: list[Relation] = []
        for r in data.get("relations", []):
            pred_str = str(r.get("predicate", "USED_SKILL")).upper()
            if pred_str in PredicateType.__members__:
                pred = PredicateType[pred_str]
            else:
                pred = PredicateType.USES_SKILL
            relations.append(
                Relation(
                    subject_id=r["subject_id"],
                    predicate=pred,
                    object_id=r["object_id"],
                    start_date=r.get("start_date"),
                    end_date=r.get("end_date"),
                    confidence=float(r.get("confidence", 1.0)),
                    source_snippet=r.get("source_snippet", ""),
                )
            )

        # Ensure candidate entity is present
        cand_found = any(e.id == candidate_id for e in entities)
        if not cand_found:
            entities.insert(
                0,
                Entity(id=candidate_id, name=name, label=name, category=EntityCategory.CANDIDATE),
            )

        return ExtractedGraph(
            candidate_id=candidate_id,
            name=name,
            entities=entities,
            relations=relations,
            raw_text=cv_text,
        )

    def _extract_with_llm_job(
        self, job_text: str, job_id: str, title: Optional[str]
    ) -> JobDescriptionGraphPayload:
        prompt = RELATION_EXTRACTION_PROMPT.format(text=job_text)
        res = self._llm.invoke(prompt)
        content = res.content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        data = json.loads(content)
        job_title = title or data.get("title") or "Target Role"

        entities = [
            Entity(
                id=e["id"],
                label=e.get("label", e["id"]),
                category=EntityCategory[e.get("category", "SKILL").upper()],
            )
            for e in data.get("entities", [])
            if e.get("category", "").upper() in EntityCategory.__members__
        ]

        relations = [
            Relation(
                subject_id=r["subject_id"],
                predicate=PredicateType[r.get("predicate", "USED_SKILL").upper()],
                object_id=r["object_id"],
                confidence=float(r.get("confidence", 1.0)),
            )
            for r in data.get("relations", [])
            if r.get("predicate", "").upper() in PredicateType.__members__
        ]

        return JobDescriptionGraphPayload(
            job_id=job_id,
            title=job_title,
            entities=entities,
            relations=relations,
            raw_text=job_text,
        )

    def _extract_deterministic_candidate(
        self, cv_text: str, candidate_id: str, candidate_name: Optional[str]
    ) -> CandidateGraphPayload:
        """
        Deterministic parser for offline processing and robust test execution.
        Extracts candidate, roles, dates, skills, and links them via dependencies.
        """
        name = candidate_name or self._guess_name(cv_text) or candidate_id
        entities: list[Entity] = [
            Entity(id=candidate_id, label=name, category=EntityCategory.CANDIDATE)
        ]
        relations: list[Relation] = []

        # Find skills present in text
        found_skills = self._detect_skills_in_text(cv_text)

        # Detect experience blocks with dates (e.g. 2018 - 2021)
        role_blocks = self._parse_experience_blocks(cv_text)

        if not role_blocks:
            # Default general role if no specific blocks found
            general_role_id = f"role:{candidate_id}_experience"
            entities.append(
                Entity(
                    id=general_role_id,
                    label="Professional Experience",
                    category=EntityCategory.ROLE,
                )
            )
            relations.append(
                Relation(
                    subject_id=candidate_id,
                    predicate=PredicateType.HELD_ROLE,
                    object_id=general_role_id,
                    confidence=0.9,
                )
            )
            for s_id, s_label in found_skills:
                entities.append(
                    Entity(id=s_id, label=s_label, category=EntityCategory.SKILL)
                )
                relations.append(
                    Relation(
                        subject_id=general_role_id,
                        predicate=PredicateType.USED_SKILL,
                        object_id=s_id,
                        confidence=0.85,
                    )
                )
        else:
            for idx, block in enumerate(role_blocks):
                role_id = f"role:{candidate_id}_{idx}"
                role_label = block["role_title"]
                entities.append(
                    Entity(id=role_id, label=role_label, category=EntityCategory.ROLE)
                )

                # Link candidate to role with dates
                relations.append(
                    Relation(
                        subject_id=candidate_id,
                        predicate=PredicateType.HELD_ROLE,
                        object_id=role_id,
                        start_date=block.get("start_date"),
                        end_date=block.get("end_date"),
                        confidence=0.95,
                    )
                )

                # Company if found
                if block.get("company"):
                    company_id = f"company:{re.sub(r'[^a-zA-Z0-9]', '', block['company']).lower()}"
                    entities.append(
                        Entity(
                            id=company_id,
                            label=block["company"],
                            category=EntityCategory.COMPANY,
                        )
                    )
                    relations.append(
                        Relation(
                            subject_id=role_id,
                            predicate=PredicateType.WORKED_AT,
                            object_id=company_id,
                            start_date=block.get("start_date"),
                            end_date=block.get("end_date"),
                            confidence=0.95,
                        )
                    )

                # Link skills detected inside this block to this role
                block_skills = self._detect_skills_in_text(block["text"])
                for s_id, s_label in block_skills:
                    if not any(e.id == s_id for e in entities):
                        entities.append(
                            Entity(id=s_id, name=s_label, label=s_label, category=EntityCategory.SKILL)
                        )
                    relations.append(
                        Relation(
                            subject_id=role_id,
                            predicate=PredicateType.USES_SKILL,
                            object_id=s_id,
                            start_date=block.get("start_date"),
                            end_date=block.get("end_date"),
                            confidence=0.90,
                            source_snippet=block.get("text", ""),
                        )
                    )

            # Add any remaining globally detected skills
            for s_id, s_label in found_skills:
                if not any(e.id == s_id for e in entities):
                    entities.append(
                        Entity(id=s_id, name=s_label, label=s_label, category=EntityCategory.SKILL)
                    )
                if not any(r.object_id == s_id for r in relations):
                    first_role_id = f"role:{candidate_id}_0"
                    relations.append(
                        Relation(
                            subject_id=first_role_id,
                            predicate=PredicateType.USES_SKILL,
                            object_id=s_id,
                            confidence=0.75,
                            source_snippet="",
                        )
                    )

        # Detect degrees
        degree_blocks = self._detect_degrees(cv_text)
        for d_id, d_label, d_date in degree_blocks:
            entities.append(
                Entity(id=d_id, name=d_label, label=d_label, category=EntityCategory.DEGREE)
            )
            relations.append(
                Relation(
                    subject_id=candidate_id,
                    predicate=PredicateType.EARNED_DEGREE,
                    object_id=d_id,
                    end_date=d_date,
                    confidence=0.95,
                    source_snippet=d_label,
                )
            )

        return ExtractedGraph(
            candidate_id=candidate_id,
            name=name,
            entities=entities,
            relations=relations,
            raw_text=cv_text,
        )

    def _extract_deterministic_job(
        self, job_text: str, job_id: str, title: Optional[str]
    ) -> JobDescriptionGraphPayload:
        job_title = title or "Target Job"
        entities: list[Entity] = [
            Entity(id=job_id, label=job_title, category=EntityCategory.ROLE)
        ]
        relations: list[Relation] = []

        found_skills = self._detect_skills_in_text(job_text)
        for s_id, s_label in found_skills:
            entities.append(
                Entity(id=s_id, label=s_label, category=EntityCategory.SKILL)
            )
            relations.append(
                Relation(
                    subject_id=job_id,
                    predicate=PredicateType.USED_SKILL,
                    object_id=s_id,
                    confidence=1.0,
                )
            )

        return JobDescriptionGraphPayload(
            job_id=job_id,
            title=job_title,
            entities=entities,
            relations=relations,
            raw_text=job_text,
        )

    def _detect_skills_in_text(self, text: str) -> list[tuple[str, str]]:
        """Identify skill keywords in text."""
        from src.core.ontology import SYNONYM_MAP

        results: list[tuple[str, str]] = []
        lower = text.lower()

        # Check all canonical skills
        for term, norm_id in SYNONYM_MAP.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lower):
                # Clean label
                label = term.title()
                if (norm_id, label) not in results:
                    results.append((norm_id, label))

        return results

    def _parse_experience_blocks(self, text: str) -> list[dict]:
        """Parse employment date ranges and associated text snippets."""
        blocks: list[dict] = []

        # Matches patterns like '2019 - 2022', '01/2020 - 05/2023', '2021 - Present'
        pattern = re.compile(
            r"(?P<start>(?:\d{1,2}[/\-])?\d{4})\s*(?:-|à|to|au)\s*(?P<end>(?:\d{1,2}[/\-])?\d{4}|present|actuel|en cours)",
            re.IGNORECASE,
        )

        matches = list(pattern.finditer(text))
        for i, match in enumerate(matches):
            start_str = match.group("start")
            end_str = match.group("end")

            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            snippet = text[start_pos:end_pos].strip()

            # First non-empty line as title
            lines = [l.strip() for l in snippet.splitlines() if l.strip()]
            role_title = lines[0] if lines else f"Experience {i+1}"

            blocks.append({
                "role_title": role_title,
                "start_date": start_str,
                "end_date": end_str,
                "text": snippet,
                "company": None,
            })

        return blocks

    def _detect_degrees(self, text: str) -> list[tuple[str, str, Optional[str]]]:
        """Detect academic degrees in text."""
        degrees: list[tuple[str, str, Optional[str]]] = []
        lower = text.lower()

        degree_terms = [
            ("degree:phd", "Doctorat / PhD", ["phd", "doctorat", "doctor"]),
            ("degree:master", "Master / Bac+5", ["master", "bac+5", "msc", "ingénieur", "diplôme d'ingénieur"]),
            ("degree:bachelor", "Licence / Bachelor", ["bachelor", "licence", "bac+3", "bsc"]),
        ]

        for d_id, label, keywords in degree_terms:
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                    degrees.append((d_id, label, None))
                    break

        return degrees

    def _guess_name(self, text: str) -> Optional[str]:
        """Heuristic candidate name detector from top lines."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line.split()) in [2, 3] and not re.search(r"[:\d]", first_line):
                return first_line
        return None

