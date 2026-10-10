"""
src/core/models.py
------------------
Pydantic schemas for structured entity, relation, and knowledge graph extraction.
Supports structured outputs from LLMs (Instructor / native JSON tool-calling).
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityCategory(str, Enum):
    """Supported entity categories for the CV/JD and technical document ontologies."""
    CANDIDATE = "CANDIDATE"
    ROLE = "ROLE"
    SKILL = "SKILL"
    COMPANY = "COMPANY"
    DEGREE = "DEGREE"
    PROJECT = "PROJECT"
    # Technical Document & Specification categories
    CONCEPT = "CONCEPT"
    SPECIFICATION = "SPECIFICATION"
    SECTION = "SECTION"
    PARAMETER = "PARAMETER"
    PROTOCOL = "PROTOCOL"


class PredicateType(str, Enum):
    """Semantic relation types connecting entities."""
    HELD_ROLE = "HELD_ROLE"
    WORKED_AT = "WORKED_AT"
    USES_SKILL = "USES_SKILL"
    USED_SKILL = "USED_SKILL"  # Synonym for USES_SKILL
    EARNED_DEGREE = "EARNED_DEGREE"
    DELIVERED_PROJECT = "DELIVERED_PROJECT"
    # Technical Document & Specification predicates
    DEPENDS_ON = "DEPENDS_ON"
    DEFINES = "DEFINES"
    CONFLICTS_WITH = "CONFLICTS_WITH"


class Entity(BaseModel):
    """
    Represents an extracted named entity grounded in the knowledge graph.
    Supports both `name` and `label` transparently.
    """
    model_config = ConfigDict(frozen=False, extra="ignore", populate_by_name=True)

    id: str = Field(..., description="Unique identifier (e.g. 'skill:pytorch')")
    name: str = Field(default="", description="Surface form or display name")
    label: str = Field(default="", description="Alias for name")
    category: EntityCategory = Field(..., description="Ontological category")
    claimed_years: Optional[float] = Field(None, description="Claimed years of experience if stated")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    @model_validator(mode="before")
    @classmethod
    def harmonize_name_label(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Normalize id
            if "id" in values and isinstance(values["id"], str):
                values["id"] = values["id"].strip().lower()

            name = values.get("name") or values.get("label") or values.get("id", "")
            if not values.get("name"):
                values["name"] = name
            if not values.get("label"):
                values["label"] = name

            # Uppercase category string if string passed
            cat = values.get("category")
            if isinstance(cat, str):
                values["category"] = cat.upper()
        return values


class Relation(BaseModel):
    """
    Represents a directed knowledge triplet (Subject-Predicate-Object)
    with temporal qualifiers, confidence score, and text source snippet.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    subject_id: str = Field(..., description="ID of source entity")
    predicate: PredicateType = Field(..., description="Predicate relation type")
    object_id: str = Field(..., description="ID of target entity")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM or YYYY or null)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM or YYYY or null)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence (0.0 to 1.0)")
    source_snippet: str = Field(default="", description="Textual source snippet supporting this relation")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contextual snippet, metadata")

    @model_validator(mode="before")
    @classmethod
    def sanitize_relation(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "subject_id" in values and isinstance(values["subject_id"], str):
                values["subject_id"] = values["subject_id"].strip().lower()
            if "object_id" in values and isinstance(values["object_id"], str):
                values["object_id"] = values["object_id"].strip().lower()

            pred = values.get("predicate")
            if isinstance(pred, str):
                pred_upper = pred.upper()
                if pred_upper == "USED_SKILL":
                    values["predicate"] = PredicateType.USES_SKILL
                elif pred_upper in PredicateType.__members__:
                    values["predicate"] = PredicateType[pred_upper]
        return values


class RequirementImportance(str, Enum):
    MUST = "must"
    NICE = "nice"


class JobRequirement(BaseModel):
    """Specific skill or competency requirement for a job posting."""
    model_config = ConfigDict(frozen=False, extra="ignore")

    skill_id: str = Field(..., description="Normalized skill ID")
    name: str = Field(default="", description="Display name of requirement")
    importance: Literal["must", "nice"] = Field(default="must", description="Must-have or nice-to-have requirement")
    weight: float = Field(default=1.0, description="Matching weight factor (e.g. 1.0 for must, 0.5 for nice)")


DEGREE_LEVEL_MAP: dict[str, int] = {
    "none": 0,
    "sans": 0,
    "bac": 1,
    "baccalaureat": 1,
    "high school": 1,
    "bac+2": 2,
    "bts": 2,
    "dut": 2,
    "deug": 2,
    "bac+3": 3,
    "licence": 3,
    "bachelor": 3,
    "but": 3,
    "bac+5": 5,
    "master": 5,
    "ingenieur": 5,
    "msc": 5,
    "dea": 5,
    "dess": 5,
    "bac+8": 8,
    "doctorat": 8,
    "phd": 8,
}


def parse_date_to_months(date_str: Optional[str]) -> Optional[int]:
    """Parses a YYYY or YYYY-MM date string into total absolute months."""
    if not date_str:
        return None
    cleaned = str(date_str).strip()
    m_ym = re.match(r"^(\d{4})[-/.](\d{1,2})", cleaned)
    if m_ym:
        return int(m_ym.group(1)) * 12 + int(m_ym.group(2))
    m_y = re.match(r"^(\d{4})", cleaned)
    if m_y:
        return int(m_y.group(1)) * 12 + 1
    return None


def merge_date_intervals_months(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merges overlapping or contiguous [start, end] month intervals."""
    valid = [iv for iv in intervals if iv[0] <= iv[1]]
    if not valid:
        return []
    sorted_ivs = sorted(valid, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for cur_s, cur_e in sorted_ivs[1:]:
        last_s, last_e = merged[-1]
        if cur_s <= last_e:
            merged[-1] = (last_s, max(last_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


class ExtractedGraph(BaseModel):
    """
    Structured container holding candidate entities and relations.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    candidate_id: str = Field(..., description="Unique candidate ID")
    name: str = Field(default="", description="Candidate name")
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    relations: list[Relation] = Field(default_factory=list, description="Extracted relations")
    raw_text: Optional[str] = Field(None, description="Raw source CV text")
    degree_level: Optional[int] = Field(None, description="Explicit degree level on standard 0-8 scale")
    experience_years: Optional[float] = Field(None, description="Explicit experience duration in years")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        target = entity_id.strip().lower()
        for e in self.entities:
            if e.id == target:
                return e
        return None

    def get_skills(self) -> list[Entity]:
        return [e for e in self.entities if e.category == EntityCategory.SKILL]

    def get_roles(self) -> list[Entity]:
        return [e for e in self.entities if e.category == EntityCategory.ROLE]

    def get_relations_for_subject(self, subject_id: str) -> list[Relation]:
        target = subject_id.strip().lower()
        return [r for r in self.relations if r.subject_id == target]

    def compute_active_career_duration_years(self) -> float:
        """
        Computes active career duration in years by parsing and merging
        overlapping employment intervals in the candidate graph.
        """
        if self.experience_years is not None:
            return float(self.experience_years)

        raw_intervals: list[tuple[int, int]] = []
        # 1. From relations
        for r in self.relations:
            if r.start_date and r.end_date:
                s_m = parse_date_to_months(r.start_date)
                e_m = parse_date_to_months(r.end_date)
                if s_m and e_m and s_m <= e_m:
                    raw_intervals.append((s_m, e_m))

        # 2. Fallback to raw text regex if relations had no parsed dates
        if not raw_intervals and self.raw_text:
            date_matches = re.findall(r"(\d{4}(?:[-/.]\d{1,2})?)\s*(?:-|à|to)\s*(\d{4}(?:[-/.]\d{1,2})?)", self.raw_text)
            for s_str, e_str in date_matches:
                s_m = parse_date_to_months(s_str)
                e_m = parse_date_to_months(e_str)
                if s_m and e_m and s_m <= e_m:
                    raw_intervals.append((s_m, e_m))

        if not raw_intervals:
            return 0.0

        merged = merge_date_intervals_months(raw_intervals)
        total_months = sum(e - s for s, e in merged)
        return round(total_months / 12.0, 2)

    def resolve_degree_level(self) -> int:
        """
        Resolves the highest academic degree level (scale 0 to 8).
        """
        if self.degree_level is not None:
            return int(self.degree_level)

        highest = 0
        # Check entities
        for e in self.entities:
            if e.category == EntityCategory.DEGREE:
                name_clean = e.name.lower()
                for key, lvl in DEGREE_LEVEL_MAP.items():
                    if key in name_clean:
                        highest = max(highest, lvl)

        # Check raw text
        if self.raw_text:
            text_lower = self.raw_text.lower()
            for key, lvl in DEGREE_LEVEL_MAP.items():
                if re.search(r"\b" + re.escape(key) + r"\b", text_lower):
                    highest = max(highest, lvl)

        return highest


# Alias CandidateGraphPayload to ExtractedGraph for backwards compatibility
class CandidateGraphPayload(ExtractedGraph):
    pass


class JobDescriptionGraphPayload(BaseModel):
    """
    Container payload holding extracted entities and relations for a job description.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    job_id: str = Field(..., description="Unique job ID")
    title: str = Field(..., description="Job title / role name")
    entities: list[Entity] = Field(default_factory=list, description="Extracted JD entities")
    relations: list[Relation] = Field(default_factory=list, description="Extracted JD relations")
    requirements: list[JobRequirement] = Field(default_factory=list, description="Explicit must/nice requirements")
    required_experience_years: Optional[float] = Field(None, description="Required years of experience")
    required_degree_level: Optional[int] = Field(None, description="Required degree level on standard 0-8 scale")
    raw_text: Optional[str] = Field(None, description="Raw job description text")

    def get_skills(self) -> list[Entity]:
        return [e for e in self.entities if e.category == EntityCategory.SKILL]

    def get_must_skills(self) -> list[str]:
        """Returns list of normalized skill IDs that are mandatory (must-have)."""
        must_list = [r.skill_id.strip().lower() for r in self.requirements if r.importance == "must"]
        if not must_list and not self.requirements:
            # If no explicit requirements list, default all JD skills to must-have
            must_list = [e.id.strip().lower() for e in self.get_skills()]
        return must_list

    def get_nice_skills(self) -> list[str]:
        """Returns list of normalized skill IDs that are optional / nice-to-have."""
        return [r.skill_id.strip().lower() for r in self.requirements if r.importance == "nice"]


class ConflictSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConflictType(str, Enum):
    TEMPORAL_INVERSION = "TEMPORAL_INVERSION"
    CUMULATIVE_OVER_CLAIMING = "CUMULATIVE_OVER_CLAIMING"
    ANACHRONISM = "ANACHRONISM"
    HALLUCINATED_ENTITY = "HALLUCINATED_ENTITY"
    # Technical Document Conflict types
    SPECIFICATION_CONFLICT = "SPECIFICATION_CONFLICT"
    CIRCULAR_DEPENDENCY = "CIRCULAR_DEPENDENCY"
    PARAMETER_MISMATCH = "PARAMETER_MISMATCH"
    DEPRECATED_DEPENDENCY = "DEPRECATED_DEPENDENCY"


class Conflict(BaseModel):
    """
    Represents an individual inconsistency or contradiction.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    conflict_type: ConflictType = Field(..., description="Category of conflict")
    severity: ConflictSeverity = Field(default=ConflictSeverity.MEDIUM, description="Severity rating")
    message: str = Field(..., description="Human-readable description")
    entity_ids: list[str] = Field(default_factory=list, description="IDs of entities involved")
    penalty_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Deduction penalty factor contribution")
    details: dict[str, Any] = Field(default_factory=dict, description="Supporting details")


class ConflictReport(BaseModel):
    """
    Conflict report containing detected anomalies and aggregate penalty score.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    conflicts: list[dict[str, Any]] = Field(default_factory=list, description="List of conflict dictionaries")
    penalty_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregate penalty score in [0.0, 1.0]")
    is_valid: bool = Field(default=True, description="True if within acceptable conflict thresholds")
    summary: str = Field(default="", description="Executive summary")

    @property
    def penalty_factor(self) -> float:
        return self.penalty_score


class ValidationReport(BaseModel):
    """
    Structured validation report compatible with ConflictReport.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    conflicts: list[Conflict] = Field(default_factory=list, description="List of detected conflicts")
    penalty_factor: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregate penalty factor in [0.0, 1.0]")
    penalty_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Alias for penalty factor")
    is_valid: bool = Field(default=True, description="True if no high-severity conflicts are present")
    summary: str = Field(default="", description="Summary of validation")

    @model_validator(mode="before")
    @classmethod
    def sync_penalty(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "penalty_factor" in values and "penalty_score" not in values:
                values["penalty_score"] = values["penalty_factor"]
            elif "penalty_score" in values and "penalty_factor" not in values:
                values["penalty_factor"] = values["penalty_score"]
        return values


class GraphMatchBreakdown(BaseModel):
    """
    Detailed breakdown of topological matching scores between CV and JD.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    ppr_score: float = Field(default=0.0, description="Personalized PageRank diffusion weight")
    ontological_distance_score: float = Field(default=0.0, description="SKOS taxonomic path similarity score")
    jaccard_similarity: float = Field(default=0.0, description="Skill/entity set Jaccard similarity")
    graph_match_score: float = Field(default=0.0, description="Synthesized topological score [0.0, 1.0]")
    matched_skills_exact: list[str] = Field(default_factory=list, description="Direct matching skills")
    matched_skills_inferred: list[dict[str, Any]] = Field(default_factory=list, description="Inferred taxonomic matches")
    missing_must_skills: list[str] = Field(default_factory=list, description="Mandatory skills missing from candidate")


class CandidateRankingResult(BaseModel):
    """
    Final ranking record including multi-factor score and complete audit trail.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    candidate_id: str
    candidate_name: str
    final_score: float = Field(..., description="Final composite score [0.0, 100.0]")
    vector_similarity: float = Field(default=0.0, description="Textual TF-IDF / embedding similarity [0.0, 1.0]")
    graph_match_score: float = Field(default=0.0, description="Topological match score [0.0, 1.0]")
    experience_score: float = Field(default=1.0, description="Seniority / experience duration match [0.0, 1.0]")
    degree_score: float = Field(default=1.0, description="Academic degree qualification score [0.0, 1.0]")
    conflict_penalty: float = Field(default=0.0, description="Deduction factor [0.0, 1.0]")
    missing_must_requirements: list[str] = Field(default_factory=list, description="List of missing mandatory skills")
    is_capped_by_must_have: bool = Field(default=False, description="Whether score was capped at 60% due to missing must-have")
    graph_breakdown: Optional[GraphMatchBreakdown] = None
    validation_report: Optional[ValidationReport] = None
    conflict_report: Optional[ConflictReport] = None
    critique_summary: Optional[str] = None
