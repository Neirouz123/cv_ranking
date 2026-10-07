"""
src/core/models.py
------------------
Pydantic schemas for structured entity, relation, and knowledge graph extraction.
Supports structured outputs from LLMs (Instructor / native JSON tool-calling).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityCategory(str, Enum):
    """Supported entity categories for the CV/JD ontology."""
    CANDIDATE = "CANDIDATE"
    ROLE = "ROLE"
    SKILL = "SKILL"
    COMPANY = "COMPANY"
    DEGREE = "DEGREE"
    PROJECT = "PROJECT"


class PredicateType(str, Enum):
    """Semantic relation types connecting entities."""
    HELD_ROLE = "HELD_ROLE"
    WORKED_AT = "WORKED_AT"
    USES_SKILL = "USES_SKILL"
    USED_SKILL = "USED_SKILL"  # Synonym for USES_SKILL
    EARNED_DEGREE = "EARNED_DEGREE"
    DELIVERED_PROJECT = "DELIVERED_PROJECT"


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
    raw_text: Optional[str] = Field(None, description="Raw job description text")

    def get_skills(self) -> list[Entity]:
        return [e for e in self.entities if e.category == EntityCategory.SKILL]


class ConflictSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConflictType(str, Enum):
    TEMPORAL_INVERSION = "TEMPORAL_INVERSION"
    CUMULATIVE_OVER_CLAIMING = "CUMULATIVE_OVER_CLAIMING"
    ANACHRONISM = "ANACHRONISM"
    HALLUCINATED_ENTITY = "HALLUCINATED_ENTITY"


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
    conflict_penalty: float = Field(default=0.0, description="Deduction factor [0.0, 1.0]")
    graph_breakdown: Optional[GraphMatchBreakdown] = None
    validation_report: Optional[ValidationReport] = None
    conflict_report: Optional[ConflictReport] = None
    critique_summary: Optional[str] = None
