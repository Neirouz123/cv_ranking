"""
src/core/models.py
------------------
Pydantic schemas for structured entity, relation, and knowledge graph payloads.
Designed for structured output extraction (e.g. Instructor, native JSON tool calling).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityCategory(str, Enum):
    """Supported entity categories for the CV/JD ontology."""
    CANDIDATE = "CANDIDATE"
    ROLE = "ROLE"
    SKILL = "SKILL"
    DEGREE = "DEGREE"
    COMPANY = "COMPANY"
    PROJECT = "PROJECT"


class PredicateType(str, Enum):
    """Semantic relation types connecting entities."""
    HELD_ROLE = "HELD_ROLE"
    WORKED_AT = "WORKED_AT"
    USED_SKILL = "USED_SKILL"
    DELIVERED_PROJECT = "DELIVERED_PROJECT"
    EARNED_DEGREE = "EARNED_DEGREE"


class Entity(BaseModel):
    """
    Represents an extracted named entity grounded in the knowledge graph.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: str = Field(..., description="Unique deterministic identifier for the entity (e.g., 'skill:pytorch')")
    label: str = Field(..., description="Surface form or canonical display name of the entity")
    category: EntityCategory = Field(..., description="Ontological category of the entity")
    claimed_years: Optional[float] = Field(None, description="Explicitly claimed years of experience if stated")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (e.g. aliases, source text)")

    @field_validator("id")
    @classmethod
    def sanitize_id(cls, v: str) -> str:
        return v.strip().lower()


class Relation(BaseModel):
    """
    Represents a directed knowledge triplet (Subject-Predicate-Object)
    with temporal qualifiers and confidence score.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    subject_id: str = Field(..., description="ID of source entity")
    predicate: PredicateType = Field(..., description="Predicate relation type")
    object_id: str = Field(..., description="ID of target entity")
    start_date: Optional[str] = Field(None, description="Temporal qualifier: start date in ISO or fuzzy format (e.g. '2020-01' or '2020')")
    end_date: Optional[str] = Field(None, description="Temporal qualifier: end date in ISO or fuzzy format (e.g. '2023-05' or 'present')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score between 0.0 and 1.0")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Contextual snippet, role tenure, etc.")

    @field_validator("subject_id", "object_id")
    @classmethod
    def sanitize_ids(cls, v: str) -> str:
        return v.strip().lower()


class CandidateGraphPayload(BaseModel):
    """
    Container payload holding extracted entities and relations for a single candidate.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    candidate_id: str = Field(..., description="Unique candidate ID")
    name: str = Field(..., description="Full candidate name")
    entities: list[Entity] = Field(default_factory=list, description="Extracted candidate entities")
    relations: list[Relation] = Field(default_factory=list, description="Extracted relational triplets")
    raw_text: Optional[str] = Field(None, description="Raw CV source text")

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


class JobDescriptionGraphPayload(BaseModel):
    """
    Container payload holding extracted entities and relations for a job description.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    job_id: str = Field(..., description="Unique job ID")
    title: str = Field(..., description="Job title / role name")
    entities: list[Entity] = Field(default_factory=list, description="Extracted JD entities")
    relations: list[Relation] = Field(default_factory=list, description="Extracted JD relational triplets")
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
    Represents an individual inconsistency or contradiction detected in candidate data.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    conflict_type: ConflictType = Field(..., description="Category of conflict detected")
    severity: ConflictSeverity = Field(default=ConflictSeverity.MEDIUM, description="Severity rating")
    message: str = Field(..., description="Human-readable description of the anomaly")
    entity_ids: list[str] = Field(default_factory=list, description="IDs of entities involved")
    penalty_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Deduction penalty factor contribution")
    details: dict[str, Any] = Field(default_factory=dict, description="Supporting details (e.g. claimed vs actual dates)")


class ValidationReport(BaseModel):
    """
    Comprehensive validation report detailing detected conflicts and aggregate penalty factor.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    conflicts: list[Conflict] = Field(default_factory=list, description="List of detected conflicts")
    penalty_factor: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregate penalty factor in [0.0, 1.0]")
    is_valid: bool = Field(default=True, description="True if no high-severity conflicts are present")
    summary: str = Field(default="", description="Executive summary of validation results")


class GraphMatchBreakdown(BaseModel):
    """
    Detailed breakdown of topological matching scores between CV and JD.
    """
    model_config = ConfigDict(frozen=False, extra="ignore")

    ppr_score: float = Field(default=0.0, description="Personalized PageRank diffusion weight on candidate/skills")
    ontological_distance_score: float = Field(default=0.0, description="SKOS taxonomic path similarity score")
    jaccard_similarity: float = Field(default=0.0, description="Skill/entity set Jaccard similarity")
    graph_match_score: float = Field(default=0.0, description="Synthesized topological score [0.0, 1.0]")
    matched_skills_exact: list[str] = Field(default_factory=list, description="Direct matching skills")
    matched_skills_inferred: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Taxonomically inferred matches: [{candidate_skill, jd_skill, distance, similarity}]"
    )


class CandidateRankingResult(BaseModel):
    """
    Final ranking record including hybrid multi-factor score and complete audit trail.
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
    critique_summary: Optional[str] = None

