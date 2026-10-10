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
    JobRequirement,
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


def get_llm_client(api_key: Optional[str] = None):
    """
    On-demand lazy-loader for ChatGroq. Returns None if no valid API key is present
    or if langchain_groq cannot be loaded.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not str(key).strip():
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model_name="llama-3.3-70b-versatile", api_key=key, temperature=0.1)
    except Exception:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=0.1)
        except Exception:
            return None


# Structured extraction prompt for Technical Documents, RFCs, and Specifications
DOCUMENT_EXTRACTION_PROMPT = """You are an expert Technical Knowledge Graph Extraction system for technical standards, RFCs, and engineering specifications.
Analyze the following text and extract a structured knowledge graph in strictly valid JSON format.

CRITICAL INSTRUCTIONS:
1. Extract all key ENTITIES:
   - CONCEPT: High-level architectural or protocol concepts (e.g., "Stream Multiplexing", "Flow Control", "HPACK Compression", "Zero-RTT")
   - SPECIFICATION: Standards, RFCs, or formal specifications (e.g., "RFC 7540", "RFC 9000", "TLS 1.2", "TCP", "MQTT 5.0")
   - PROTOCOL: Concrete protocols or wire formats (e.g., "HTTP/2", "QUIC", "HTTP/1.1", "h2c")
   - SECTION: Specific sections or clauses in the document (e.g., "Section 3.1", "Section 5.2 - Flow Control")
   - PARAMETER: Configurable parameters, settings, or constants (e.g., "SETTINGS_INITIAL_WINDOW_SIZE", "KeepAlive", "MaxConcurrentStreams")
2. Extract RELATIONS with confidence and verbatim source snippets:
   - CONCEPT -> DEPENDS_ON -> SPECIFICATION
   - SECTION -> DEFINES -> PARAMETER (include defined value in metadata if applicable)
   - PROTOCOL -> CONFLICTS_WITH -> PROTOCOL (incompatible protocols, cipher suites, or mutual exclusions)
   - PROTOCOL -> DEPENDS_ON -> SPECIFICATION
3. DO NOT hallucinate any entity or claim not explicitly supported by the text.

JSON Output Schema:
{{
  "document_title": "<document title or id>",
  "entities": [
    {{
      "id": "<category_prefix:identifier>",
      "name": "<display name>",
      "category": "CONCEPT" | "SPECIFICATION" | "PROTOCOL" | "SECTION" | "PARAMETER",
      "metadata": {{"value": "<parameter value or details>"}}
    }}
  ],
  "relations": [
    {{
      "subject_id": "<entity_id>",
      "predicate": "DEPENDS_ON" | "DEFINES" | "CONFLICTS_WITH",
      "object_id": "<entity_id>",
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
    Extracts structured entities, relations, and temporal qualifiers from CV and JD documents,
    as well as technical papers, RFCs, and engineering specifications.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        use_llm: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.use_llm = use_llm
        self.api_key = api_key
        self._llm = None

        if self.use_llm:
            self._llm = get_llm_client(api_key=api_key)

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

    def extract_document(
        self,
        document_text: str,
        document_id: Optional[str] = None,
        document_title: Optional[str] = None,
    ) -> ExtractedGraph:
        """
        Extract knowledge graph representation for a technical document, RFC, or specification.
        """
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:6]}"
        title = document_title or "Technical Document"

        if self._llm is not None:
            try:
                return self._extract_with_llm_document(document_text, doc_id, title)
            except Exception:
                pass

        return self._extract_deterministic_document(document_text, doc_id, title)

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

        cand_graph = ExtractedGraph(
            candidate_id=candidate_id,
            name=name,
            entities=entities,
            relations=relations,
            raw_text=cv_text,
        )
        cand_graph.degree_level = cand_graph.resolve_degree_level()
        cand_graph.experience_years = cand_graph.compute_active_career_duration_years()
        return cand_graph

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

        # Extract required experience years
        lower_job = job_text.lower()
        req_exp = None
        range_m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:à|to|-)\s*(\d+(?:[.,]\d+)?)\s*(?:ans|annees|années|years)", lower_job)
        if range_m:
            req_exp = (float(range_m.group(1).replace(",", ".")) + float(range_m.group(2).replace(",", "."))) / 2.0
        else:
            single_m = re.search(r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:ans|annees|années|years)", lower_job)
            if single_m:
                req_exp = float(single_m.group(1).replace(",", "."))
            elif re.search(r"\b(?:senior|sénior|lead)\b", lower_job):
                req_exp = 5.0
            elif re.search(r"\b(?:junior|debutant|débutant)\b", lower_job):
                req_exp = 1.5
            elif re.search(r"\b(?:stage|internship|intern)\b", lower_job):
                req_exp = 0.5

        # Extract required degree level
        req_deg = None
        if re.search(r"\b(?:bac\+8|doctorat|phd)\b", lower_job):
            req_deg = 8
        elif re.search(r"\b(?:bac\+5|master|ingenieur|ingénieur|msc)\b", lower_job):
            req_deg = 5
        elif re.search(r"\b(?:bac\+3|licence|bachelor|but)\b", lower_job):
            req_deg = 3
        elif re.search(r"\b(?:bac\+2|bts|dut)\b", lower_job):
            req_deg = 2
        elif re.search(r"\b(?:bac)\b", lower_job):
            req_deg = 1

        # Must vs Nice requirements detection
        sentences = re.split(r"[.\n;]", job_text)
        nice_kw = ["souhaité", "souhaite", "un plus", "nice-to-have", "nice to have", "apprécié", "apprecie", "atout", "optionnel", "bonus"]
        must_kw = ["indispensable", "requis", "must-have", "must have", "obligatoire", "exigé", "exige", "impératif", "imperatif", "incontournable"]

        requirements: list[JobRequirement] = []
        for s_id, s_label in found_skills:
            is_nice = False
            for sent in sentences:
                sent_l = sent.lower()
                if s_label.lower() in sent_l:
                    if any(kw in sent_l for kw in nice_kw) and not any(kw in sent_l for kw in must_kw):
                        is_nice = True
                        break

            importance = "nice" if is_nice else "must"
            weight = 0.5 if is_nice else 1.0
            requirements.append(JobRequirement(skill_id=s_id, name=s_label, importance=importance, weight=weight))

        return JobDescriptionGraphPayload(
            job_id=job_id,
            title=job_title,
            entities=entities,
            relations=relations,
            requirements=requirements,
            required_experience_years=req_exp,
            required_degree_level=req_deg,
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

        # Matches patterns like '2019 - 2022', '2024-06 - 2021-01', '01/2020 - 05/2023', '2021 - Present'
        pattern = re.compile(
            r"(?P<start>(?:\d{4}(?:[-/.]\d{1,2})?)|(?:\d{1,2}[-/.]\d{4}))\s*(?:-|à|to|au)\s*(?P<end>(?:\d{4}(?:[-/.]\d{1,2})?)|(?:\d{1,2}[-/.]\d{4})|present|actuel|en cours)",
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

    def _extract_with_llm_document(
        self, text: str, document_id: str, title: Optional[str]
    ) -> ExtractedGraph:
        prompt = DOCUMENT_EXTRACTION_PROMPT.format(text=text)
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
        doc_title = title or data.get("document_title") or "Technical Document"

        entities = [
            Entity(
                id=e["id"],
                label=e.get("name", e.get("label", e["id"])),
                name=e.get("name", e.get("label", e["id"])),
                category=EntityCategory[e.get("category", "CONCEPT").upper()],
                metadata=e.get("metadata", {}),
            )
            for e in data.get("entities", [])
            if e.get("category", "").upper() in EntityCategory.__members__
        ]

        relations = [
            Relation(
                subject_id=r["subject_id"],
                predicate=PredicateType[r.get("predicate", "DEPENDS_ON").upper()],
                object_id=r["object_id"],
                confidence=float(r.get("confidence", 1.0)),
                source_snippet=r.get("source_snippet", ""),
                metadata=r.get("metadata", {}),
            )
            for r in data.get("relations", [])
            if r.get("predicate", "").upper() in PredicateType.__members__
        ]

        return ExtractedGraph(
            candidate_id=document_id,
            name=doc_title,
            entities=entities,
            relations=relations,
            raw_text=text,
        )

    def _extract_deterministic_document(
        self, text: str, document_id: str, title: Optional[str]
    ) -> ExtractedGraph:
        entities_dict: dict[str, Entity] = {}
        relations: list[Relation] = []

        lines = text.splitlines()
        current_section_id = f"sec_{document_id}_main"
        entities_dict[current_section_id] = Entity(
            id=current_section_id,
            name=title or "General Document Specification",
            category=EntityCategory.SECTION,
        )

        known_entities_catalog: dict[str, tuple[EntityCategory, str, str]] = {
            "RFC 7540": (EntityCategory.SPECIFICATION, "spec:rfc7540", "RFC 7540 (HTTP/2 Standard)"),
            "RFC 9000": (EntityCategory.SPECIFICATION, "spec:rfc9000", "RFC 9000 (QUIC Transport)"),
            "RFC 7230": (EntityCategory.SPECIFICATION, "spec:rfc7230", "RFC 7230 (HTTP/1.1 Message Syntax)"),
            "HTTP/2": (EntityCategory.PROTOCOL, "proto:http2", "HTTP/2 Protocol"),
            "HTTP/1.1": (EntityCategory.PROTOCOL, "proto:http1_1", "HTTP/1.1 Protocol"),
            "HTTP/3": (EntityCategory.PROTOCOL, "proto:http3", "HTTP/3 Protocol"),
            "TLS 1.3": (EntityCategory.SPECIFICATION, "spec:tls1_3", "TLS 1.3 Security Specification"),
            "TLS 1.2": (EntityCategory.SPECIFICATION, "spec:tls1_2", "TLS 1.2 Security Specification"),
            "TLS 1.0": (EntityCategory.PROTOCOL, "proto:tls1_0", "Legacy TLS 1.0 Protocol"),
            "TCP": (EntityCategory.SPECIFICATION, "spec:tcp", "Transmission Control Protocol (TCP)"),
            "UDP": (EntityCategory.SPECIFICATION, "spec:udp", "User Datagram Protocol (UDP)"),
            "QUIC": (EntityCategory.PROTOCOL, "proto:quic", "QUIC Transport Protocol"),
            "MQTT 5.0": (EntityCategory.SPECIFICATION, "spec:mqtt5", "MQTT 5.0 OASIS Standard"),
            "MQTT 3.1.1": (EntityCategory.PROTOCOL, "proto:mqtt311", "Legacy MQTT 3.1.1"),
            "MQTT": (EntityCategory.PROTOCOL, "proto:mqtt", "MQTT Protocol"),
            "IEEE 802.11": (EntityCategory.SPECIFICATION, "spec:ieee802_11", "IEEE 802.11 Wireless Standard"),
            "ALPN": (EntityCategory.CONCEPT, "concept:alpn", "Application-Layer Protocol Negotiation (ALPN)"),
            "HPACK": (EntityCategory.CONCEPT, "concept:hpack", "HPACK Header Compression"),
            "h2c": (EntityCategory.PROTOCOL, "proto:h2c", "HTTP/2 Cleartext (h2c)"),
            "TLS Mandatory": (EntityCategory.SPECIFICATION, "spec:tls_mandatory", "TLS Mandatory Deployment Profiles"),
            "Strict Transport Security": (EntityCategory.SPECIFICATION, "spec:hsts", "Strict Transport Security (HSTS)"),
            "RFC 7540 Cipher Suite Blacklist": (EntityCategory.SPECIFICATION, "spec:rfc7540_blacklist", "RFC 7540 Cipher Suite Blacklist"),
            "RFC 7540 Blacklist": (EntityCategory.SPECIFICATION, "spec:rfc7540_blacklist", "RFC 7540 Cipher Suite Blacklist"),
            "Cipher Suite Blacklist": (EntityCategory.SPECIFICATION, "spec:rfc7540_blacklist", "RFC 7540 Cipher Suite Blacklist"),
            "CleanStart": (EntityCategory.CONCEPT, "concept:clean_start", "MQTT 5.0 CleanStart Flag"),
            "CleanSession": (EntityCategory.CONCEPT, "concept:clean_session", "MQTT 3.1.1 CleanSession Flag"),
            "Stream Multiplexing": (EntityCategory.CONCEPT, "concept:stream_multiplexing", "Stream Multiplexing"),
            "Flow Control": (EntityCategory.CONCEPT, "concept:flow_control", "Flow Control"),
            "Binary Framing": (EntityCategory.CONCEPT, "concept:binary_framing", "Binary Framing"),
            "Header Compression": (EntityCategory.CONCEPT, "concept:header_compression", "Header Compression"),
            "Zero-RTT Handshake": (EntityCategory.CONCEPT, "concept:zero_rtt", "Zero-RTT Handshake"),
            "Dynamic Congestion Control": (EntityCategory.CONCEPT, "concept:dynamic_congestion_control", "Dynamic Congestion Control"),
            "Explicit Congestion Notification": (EntityCategory.SPECIFICATION, "spec:ecn", "Explicit Congestion Notification (ECN)"),
            "Packet Identifier State Storage": (EntityCategory.SPECIFICATION, "spec:packet_id_storage", "Packet Identifier State Storage"),
            "Wireless Mesh Architecture": (EntityCategory.CONCEPT, "concept:wireless_mesh", "Wireless Mesh Architecture"),
            "QoS Exactly-Once Delivery": (EntityCategory.CONCEPT, "concept:qos2", "QoS Exactly-Once Delivery (QoS 2)"),
        }
        sorted_keys = sorted(known_entities_catalog.keys(), key=len, reverse=True)

        def _find_entity_info(text_fragment: str) -> Optional[tuple[EntityCategory, str, str]]:
            frag_lower = text_fragment.lower()
            for k in sorted_keys:
                if k.lower() in frag_lower:
                    return known_entities_catalog[k]
            return None

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            sec_match = re.search(r"(?:Section\s+(\d+(?:\.\d+)*)|(\d+\.\d+(?:\.\d+)*))\s*[:\.\-]?\s*([A-Za-z0-9_\-\s]+)?", trimmed, re.IGNORECASE)
            if sec_match and (trimmed.lower().startswith("section") or re.match(r"^\d+\.\d+", trimmed)):
                num = sec_match.group(1) or sec_match.group(2)
                stitle = (sec_match.group(3) or "").strip()
                s_clean = f"sec_{num.replace('.', '_')}"
                s_name = f"Section {num}" + (f" - {stitle}" if stitle else "")
                current_section_id = s_clean
                entities_dict[s_clean] = Entity(id=s_clean, name=s_name, category=EntityCategory.SECTION)

            param_match = re.search(
                r"(?:defines|sets|configures)?\s*([A-Za-z0-9_]{3,35})\s*(?:=|is|=:|\:)\s*([0-9]+(?:\s*[a-zA-Z]+)?)",
                trimmed,
                re.IGNORECASE,
            )
            if param_match:
                p_raw = param_match.group(1).strip()
                v_raw = param_match.group(2).strip()
                if p_raw.lower() not in {"http", "section", "rfc", "tcp", "udp", "tls", "version", "and", "the", "with"}:
                    p_id = f"param_{p_raw.lower()}"
                    entities_dict[p_id] = Entity(
                        id=p_id,
                        name=p_raw,
                        category=EntityCategory.PARAMETER,
                        metadata={"value": v_raw},
                    )
                    relations.append(
                        Relation(
                            subject_id=current_section_id,
                            predicate=PredicateType.DEFINES,
                            object_id=p_id,
                            confidence=0.95,
                            source_snippet=trimmed,
                            metadata={"value": v_raw},
                        )
                    )

            for k in sorted_keys:
                if k.lower() in trimmed.lower():
                    cat, ent_id, disp_name = known_entities_catalog[k]
                    if ent_id not in entities_dict:
                        entities_dict[ent_id] = Entity(id=ent_id, name=disp_name, category=cat)

            dep_pattern = re.compile(
                r"([A-Za-z0-9\/\.\s_\-]{2,45})\s+(?:depends on|relies on|requires|built on top of|runs over|is layered upon)\s+([A-Za-z0-9\/\.\s_\-]{2,45})",
                re.IGNORECASE,
            )
            for m in dep_pattern.finditer(trimmed):
                raw_subj = m.group(1).strip()
                raw_obj = m.group(2).strip()

                subj_info = _find_entity_info(raw_subj)
                obj_info = _find_entity_info(raw_obj)

                if subj_info and obj_info and subj_info[1] != obj_info[1]:
                    relations.append(
                        Relation(
                            subject_id=subj_info[1],
                            predicate=PredicateType.DEPENDS_ON,
                            object_id=obj_info[1],
                            confidence=0.92,
                            source_snippet=trimmed,
                        )
                    )

            conf_pattern = re.compile(
                r"([A-Za-z0-9\/\.\s_\-]{2,45})\s+(?:conflicts with|is incompatible with|forbids|cannot be negotiated with|disallows)\s+([A-Za-z0-9\/\.\s_\-]{2,55})",
                re.IGNORECASE,
            )
            for m in conf_pattern.finditer(trimmed):
                raw_a = m.group(1).strip()
                raw_b = m.group(2).strip()

                subj_info = _find_entity_info(raw_a)
                obj_info = _find_entity_info(raw_b)

                id_a = subj_info[1] if subj_info else f"proto:{re.sub(r'[^a-z0-9_]+', '_', raw_a.lower()).strip('_')}"
                id_b = obj_info[1] if obj_info else f"proto:{re.sub(r'[^a-z0-9_]+', '_', raw_b.lower()).strip('_')}"

                if not subj_info and id_a not in entities_dict:
                    entities_dict[id_a] = Entity(id=id_a, name=raw_a, category=EntityCategory.PROTOCOL)

                if not obj_info and id_b not in entities_dict:
                    entities_dict[id_b] = Entity(id=id_b, name=raw_b, category=EntityCategory.PROTOCOL)

                if id_a and id_b and id_a != id_b:
                    relations.append(
                        Relation(
                            subject_id=id_a,
                            predicate=PredicateType.CONFLICTS_WITH,
                            object_id=id_b,
                            confidence=0.95,
                            source_snippet=trimmed,
                        )
                    )

        unique_rels: list[Relation] = []
        seen = set()
        for r in relations:
            key = (r.subject_id, r.predicate, r.object_id, r.metadata.get("value"))
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)

        return ExtractedGraph(
            candidate_id=document_id,
            name=title or "Technical Document",
            entities=list(entities_dict.values()),
            relations=unique_rels,
            raw_text=text,
        )


