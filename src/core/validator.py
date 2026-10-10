"""
src/core/validator.py
---------------------
Consistency and contradiction detection engine for candidate knowledge graphs.
Implements automated detection of:
1. Temporal Inversions (start_date > end_date)
2. Cumulative Over-claiming (claimed years > documented job durations)
3. Anachronisms (technology usage predating its official release)
4. Hallucinated Entities (entities missing from CV source text)
Outputs a structured ValidationReport with conflict details and penalty factor.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser

from src.core.models import (
    CandidateGraphPayload,
    Conflict,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    EntityCategory,
    ExtractedGraph,
    PredicateType,
    ValidationReport,
)
from src.core.ontology import normalize_skill_name

# Registry of official release years for common technologies
TECH_RELEASE_REGISTRY: dict[str, int] = {
    "skill:fastapi": 2018,
    "skill:pytorch": 2016,
    "skill:tensorflow": 2015,
    "skill:kubernetes": 2014,
    "skill:docker": 2013,
    "skill:react": 2013,
    "skill:vue": 2014,
    "skill:nextjs": 2016,
    "skill:typescript": 2012,
    "skill:rust": 2010,
    "skill:go": 2009,
    "skill:swift": 2014,
    "skill:kotlin": 2011,
    "skill:transformers": 2018,
    "skill:huggingface": 2016,
    "skill:langchain": 2022,
    "skill:graphrag": 2024,
    "skill:polars": 2020,
    "skill:scikitlearn": 2007,
    "skill:pandas": 2008,
    "skill:spark": 2010,
    "skill:kafka": 2011,
    "skill:snowflake": 2014,
    "skill:mongodb": 2009,
    "skill:redis": 2009,
    "skill:nodejs": 2009,
    "skill:springboot": 2014,
}

# Current reference date for calculating ongoing tenures
CURRENT_YEAR = 2026
CURRENT_DATE = datetime(CURRENT_YEAR, 10, 1)


def parse_fuzzy_date(date_str: Optional[str], is_end_date: bool = False) -> Optional[datetime]:
    """
    Fuzzy date parser supporting various date formats, years, and present-tense markers.
    """
    if not date_str:
        return None

    cleaned = str(date_str).strip().lower()
    if not cleaned or cleaned in {"null", "none", "n/a", "unknown"}:
        return None

    # Handle present / ongoing markers
    if any(marker in cleaned for marker in ["present", "actuel", "aujourd'hui", "current", "now", "en cours"]):
        return CURRENT_DATE

    # Pure 4-digit year match (e.g. '2021')
    year_match = re.fullmatch(r"(\d{4})", cleaned)
    if year_match:
        year = int(year_match.group(1))
        # If it's an end date, default to end of that year (Dec 31)
        month = 12 if is_end_date else 1
        day = 31 if is_end_date else 1
        return datetime(year, month, day)

    # Month/Year formats e.g. 05/2021 or 2021-05
    m_y_match = re.search(r"(\d{1,2})[/\-](\d{4})", cleaned)
    if m_y_match:
        month = int(m_y_match.group(1))
        year = int(m_y_match.group(2))
        if 1 <= month <= 12:
            day = 28 if is_end_date else 1
            return datetime(year, month, day)

    y_m_match = re.search(r"(\d{4})[/\-](\d{1,2})", cleaned)
    if y_m_match:
        year = int(y_m_match.group(1))
        month = int(y_m_match.group(2))
        if 1 <= month <= 12:
            day = 28 if is_end_date else 1
            return datetime(year, month, day)

    # General dateutil fuzzy parsing
    try:
        dt = date_parser.parse(cleaned, fuzzy=True, default=datetime(2000, 1, 1))
        return dt
    except (ValueError, OverflowError):
        # Fallback: search for any 4-digit year inside
        fallback_year = re.search(r"(19\d{2}|20\d{2})", cleaned)
        if fallback_year:
            year = int(fallback_year.group(1))
            return datetime(year, 12 if is_end_date else 1, 1)
        return None


def merge_date_intervals(intervals: list[tuple[datetime, datetime]]) -> float:
    """
    Merge overlapping date intervals and calculate total cumulative duration in years.
    """
    if not intervals:
        return 0.0

    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = []

    for start, end in sorted_intervals:
        if end < start:
            continue
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    total_days = sum((end - start).days for start, end in merged)
    return total_days / 365.25


class ConsistencyValidator:
    """
    Validates candidate knowledge graph payloads against temporal and factual constraints.
    """

    def __init__(
        self,
        tech_release_registry: Optional[dict[str, int]] = None,
        max_penalty: float = 1.0,
    ) -> None:
        self.registry = tech_release_registry or TECH_RELEASE_REGISTRY
        self.max_penalty = max_penalty

    def validate(self, payload: CandidateGraphPayload) -> ValidationReport:
        """
        Run all consistency and contradiction checks on the candidate payload.
        """
        conflicts: list[Conflict] = []

        # 1. Temporal Inversion check
        conflicts.extend(self._check_temporal_inversions(payload))

        # 2. Anachronism check
        conflicts.extend(self._check_anachronisms(payload))

        # 3. Cumulative Over-claiming check
        conflicts.extend(self._check_cumulative_overclaiming(payload))

        # 4. Text boundary / hallucination check
        if payload.raw_text:
            conflicts.extend(self._check_hallucinations(payload))

        # Compute aggregate penalty factor
        penalty_factor = 0.0
        for c in conflicts:
            penalty_factor += c.penalty_weight

        penalty_factor = min(self.max_penalty, max(0.0, penalty_factor))
        is_valid = not any(c.severity == ConflictSeverity.HIGH for c in conflicts) and penalty_factor < 0.6

        summary = (
            f"Validation passed with {len(conflicts)} warning(s)."
            if is_valid and not conflicts
            else f"Validation detected {len(conflicts)} conflict(s) (Penalty factor: {penalty_factor:.2f})."
        )

        return ValidationReport(
            conflicts=conflicts,
            penalty_factor=penalty_factor,
            penalty_score=penalty_factor,
            is_valid=is_valid,
            summary=summary,
        )

    def validate_conflicts(self, payload: ExtractedGraph | CandidateGraphPayload) -> ConflictReport:
        """
        Validate candidate payload and return a ConflictReport(conflicts: list[dict], penalty_score: float).
        """
        report = self.validate(payload)
        conf_dicts = [
            {
                "conflict_type": c.conflict_type.value,
                "severity": c.severity.value,
                "message": c.message,
                "entity_ids": c.entity_ids,
                "penalty_weight": c.penalty_weight,
                "details": c.details,
            }
            for c in report.conflicts
        ]
        return ConflictReport(
            conflicts=conf_dicts,
            penalty_score=report.penalty_factor,
            is_valid=report.is_valid,
            summary=report.summary,
        )

    def _check_temporal_inversions(self, payload: CandidateGraphPayload) -> list[Conflict]:
        """Detect start_date > end_date on relations."""
        conflicts: list[Conflict] = []

        for rel in payload.relations:
            if not rel.start_date or not rel.end_date:
                continue

            start_dt = parse_fuzzy_date(rel.start_date, is_end_date=False)
            end_dt = parse_fuzzy_date(rel.end_date, is_end_date=True)

            if start_dt and end_dt and start_dt > end_dt:
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.TEMPORAL_INVERSION,
                        severity=ConflictSeverity.HIGH,
                        message=(
                            f"Temporal inversion detected for relation '{rel.subject_id} -{rel.predicate}-> "
                            f"{rel.object_id}': start_date ({rel.start_date}) is after end_date ({rel.end_date})."
                        ),
                        entity_ids=[rel.subject_id, rel.object_id],
                        penalty_weight=0.30,
                        details={
                            "start_date": rel.start_date,
                            "end_date": rel.end_date,
                            "parsed_start": start_dt.isoformat(),
                            "parsed_end": end_dt.isoformat(),
                            "predicate": rel.predicate.value,
                        },
                    )
                )

        return conflicts

    def _check_anachronisms(self, payload: CandidateGraphPayload) -> list[Conflict]:
        """Compare the earliest claimed usage date of a technology against official release date."""
        conflicts: list[Conflict] = []

        # Find all relations mentioning a skill
        for rel in payload.relations:
            skill_id = None
            if rel.predicate in {PredicateType.USED_SKILL, PredicateType.USES_SKILL}:
                skill_id = normalize_skill_name(rel.object_id)
            elif rel.subject_id.startswith("skill:"):
                skill_id = normalize_skill_name(rel.subject_id)
            elif rel.object_id.startswith("skill:"):
                skill_id = normalize_skill_name(rel.object_id)

            if not skill_id or skill_id not in self.registry:
                continue

            release_year = self.registry[skill_id]

            # Determine claimed start year
            start_year = None
            if rel.start_date:
                start_dt = parse_fuzzy_date(rel.start_date, is_end_date=False)
                if start_dt:
                    start_year = start_dt.year

            if start_year and start_year < release_year:
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.ANACHRONISM,
                        severity=ConflictSeverity.HIGH,
                        message=(
                            f"Anachronism detected: claimed usage of '{skill_id}' in {start_year}, "
                            f"but technology was officially released in {release_year}."
                        ),
                        entity_ids=[rel.subject_id, rel.object_id],
                        penalty_weight=0.35,
                        details={
                            "technology": skill_id,
                            "claimed_year": start_year,
                            "release_year": release_year,
                            "relation": f"{rel.subject_id} -> {rel.object_id}",
                        },
                    )
                )

        return conflicts

    def _check_cumulative_overclaiming(self, payload: CandidateGraphPayload) -> list[Conflict]:
        """
        Flag candidates claiming N years of specific skill experience when the total duration
        of documented jobs/roles linked to that skill is < N.
        """
        conflicts: list[Conflict] = []

        # Map role entities to their date intervals
        role_intervals: dict[str, tuple[datetime, datetime]] = {}
        for rel in payload.relations:
            if rel.predicate in {PredicateType.HELD_ROLE, PredicateType.WORKED_AT}:
                start_dt = parse_fuzzy_date(rel.start_date, is_end_date=False)
                end_dt = parse_fuzzy_date(rel.end_date, is_end_date=True) or CURRENT_DATE
                if start_dt and end_dt and start_dt <= end_dt:
                    role_intervals[rel.object_id] = (start_dt, end_dt)
                    role_intervals[rel.subject_id] = (start_dt, end_dt)

        # For each skill entity, examine explicitly claimed years vs documented tenures
        for entity in payload.get_skills():
            norm_id = normalize_skill_name(entity.id)
            claimed_years = entity.claimed_years

            # Also check metadata for claimed years
            if claimed_years is None and "claimed_years" in entity.metadata:
                try:
                    claimed_years = float(entity.metadata["claimed_years"])
                except (ValueError, TypeError):
                    pass

            if claimed_years is None or claimed_years <= 0:
                continue

            # Gather all documented intervals linked to this skill
            intervals: list[tuple[datetime, datetime]] = []

            for rel in payload.relations:
                rel_skill = normalize_skill_name(rel.object_id)
                if rel.predicate in {PredicateType.USED_SKILL, PredicateType.USES_SKILL} and rel_skill == norm_id:
                    # 1. Interval directly on the USED_SKILL relation
                    r_start = parse_fuzzy_date(rel.start_date, is_end_date=False)
                    r_end = parse_fuzzy_date(rel.end_date, is_end_date=True) or (CURRENT_DATE if r_start else None)
                    if r_start and r_end and r_start <= r_end:
                        intervals.append((r_start, r_end))
                    # 2. Or interval inherited from the role/project subject
                    elif rel.subject_id in role_intervals:
                        intervals.append(role_intervals[rel.subject_id])

            documented_years = merge_date_intervals(intervals)

            # If documented experience is significantly lower than claimed experience (margin: 0.5 years)
            if documented_years < (claimed_years - 0.5):
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.CUMULATIVE_OVER_CLAIMING,
                        severity=ConflictSeverity.MEDIUM,
                        message=(
                            f"Cumulative over-claiming detected for '{entity.label}': candidate claims "
                            f"{claimed_years:.1f} years, but documented tenures only account for "
                            f"{documented_years:.1f} years."
                        ),
                        entity_ids=[entity.id],
                        penalty_weight=0.25,
                        details={
                            "skill": entity.label,
                            "claimed_years": claimed_years,
                            "documented_years": round(documented_years, 2),
                        },
                    )
                )

        return conflicts

    def _check_hallucinations(self, payload: CandidateGraphPayload) -> list[Conflict]:
        """Ensure extracted entities appear in explicit text boundaries."""
        conflicts: list[Conflict] = []
        raw_text_lower = (payload.raw_text or "").lower()

        if not raw_text_lower:
            return conflicts

        for entity in payload.entities:
            # Check label
            label_lower = entity.label.strip().lower()
            # Canonical without prefix
            id_stem = entity.id.split(":")[-1].lower()

            found = label_lower in raw_text_lower or id_stem in raw_text_lower
            if not found and entity.metadata.get("aliases"):
                found = any(str(alias).lower() in raw_text_lower for alias in entity.metadata["aliases"])

            if not found and entity.category in {EntityCategory.SKILL, EntityCategory.COMPANY, EntityCategory.DEGREE}:
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.HALLUCINATED_ENTITY,
                        severity=ConflictSeverity.LOW,
                        message=f"Entity '{entity.label}' ({entity.category.value}) could not be verified in the source text.",
                        entity_ids=[entity.id],
                        penalty_weight=0.08,
                        details={"label": entity.label, "category": entity.category.value},
                    )
                )

        return conflicts


def validate_candidate_graph(payload: ExtractedGraph | CandidateGraphPayload) -> ConflictReport:
    """
    Validate a candidate graph and return a ConflictReport detailing
    temporal inversions, tech release anachronisms, and duration mismatches.
    """
    validator = ConsistencyValidator()
    return validator.validate_conflicts(payload)


def validate_document_graph(payload: ExtractedGraph) -> ConflictReport:
    """
    Validate a technical specification / standard document graph for:
    1. Protocol Incompatibilities (CONFLICTS_WITH relations)
    2. Circular Constraints / Dependencies (A -> DEPENDS_ON -> B and B -> DEPENDS_ON -> A)
    3. Conflicting Parameter Definitions across sections
    """
    conflicts: list[Conflict] = []

    # 1. Check explicit CONFLICTS_WITH relations
    for rel in payload.relations:
        if rel.predicate == PredicateType.CONFLICTS_WITH:
            e_subj = payload.get_entity(rel.subject_id)
            e_obj = payload.get_entity(rel.object_id)
            subj_name = e_subj.name if e_subj else rel.subject_id
            obj_name = e_obj.name if e_obj else rel.object_id
            conflicts.append(
                Conflict(
                    conflict_type=ConflictType.SPECIFICATION_CONFLICT,
                    severity=ConflictSeverity.HIGH,
                    message=f"Specification incompatibility: '{subj_name}' explicitly conflicts with '{obj_name}'.",
                    entity_ids=[rel.subject_id, rel.object_id],
                    penalty_weight=0.35,
                    details={
                        "subject": subj_name,
                        "object": obj_name,
                        "source_snippet": rel.source_snippet,
                    },
                )
            )

    # 2. Check Circular Dependencies among DEPENDS_ON relations
    dep_edges = []
    for rel in payload.relations:
        if rel.predicate == PredicateType.DEPENDS_ON:
            dep_edges.append((rel.subject_id, rel.object_id))

    if dep_edges:
        import networkx as nx
        dg = nx.DiGraph()
        dg.add_edges_from(dep_edges)
        try:
            cycles = list(nx.simple_cycles(dg))
            for cycle in cycles:
                cycle_names = []
                for cid in cycle:
                    ent = payload.get_entity(cid)
                    cycle_names.append(ent.name if ent else cid)
                cycle_str = " ➔ ".join(cycle_names) + f" ➔ {cycle_names[0]}"
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                        severity=ConflictSeverity.HIGH,
                        message=f"Circular constraint detected across specifications: {cycle_str}.",
                        entity_ids=list(cycle),
                        penalty_weight=0.30,
                        details={"cycle": cycle, "cycle_names": cycle_names},
                    )
                )
        except Exception:
            pass

    # 3. Check Conflicting Parameter Definitions across sections
    param_defs: dict[str, list[dict[str, Any]]] = {}
    for rel in payload.relations:
        if rel.predicate == PredicateType.DEFINES:
            param_ent = payload.get_entity(rel.object_id)
            sec_ent = payload.get_entity(rel.subject_id)
            p_name = param_ent.name if param_ent else rel.object_id
            p_clean = p_name.strip().upper()
            sec_name = sec_ent.name if sec_ent else rel.subject_id
            val = (
                rel.metadata.get("value")
                or (param_ent.metadata.get("value") if param_ent else None)
                or ""
            )
            if p_clean not in param_defs:
                param_defs[p_clean] = []
            param_defs[p_clean].append({
                "section": sec_name,
                "section_id": rel.subject_id,
                "param_id": rel.object_id,
                "value": str(val).strip(),
                "snippet": rel.source_snippet,
            })

    for p_name, defs in param_defs.items():
        if len(defs) > 1:
            values = {d["value"] for d in defs if d["value"]}
            if len(values) > 1:
                sec_list = ", ".join(f"{d['section']} ({d['value']})" for d in defs)
                conflicts.append(
                    Conflict(
                        conflict_type=ConflictType.PARAMETER_MISMATCH,
                        severity=ConflictSeverity.MEDIUM,
                        message=f"Contradictory parameter specification for '{p_name}': defined with conflicting values across sections: {sec_list}.",
                        entity_ids=[d["section_id"] for d in defs] + [defs[0]["param_id"]],
                        penalty_weight=0.25,
                        details={
                            "parameter": p_name,
                            "definitions": defs,
                            "distinct_values": list(values),
                        },
                    )
                )

    total_penalty = min(sum(c.penalty_weight for c in conflicts), 1.0)
    is_valid = len(conflicts) == 0

    if not conflicts:
        summary = "Document specification audit clean: no protocol conflicts, circular constraints, or parameter mismatches."
    else:
        summary = f"Detected {len(conflicts)} specification anomalies with aggregate penalty factor {total_penalty:.2f}."

    return ConflictReport(
        conflicts=[c.model_dump() for c in conflicts],
        penalty_score=round(total_penalty, 3),
        is_valid=is_valid,
        summary=summary,
    )



