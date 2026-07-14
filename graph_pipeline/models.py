from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExtractionSource(str, Enum):
    RULE_BASED = "rule_based"
    LLM_INFERRED = "llm_inferred"
    PHANTOM = "phantom"


@dataclass
class Node:
    id: str                        # namespaced: "{dataset_id}:{uniqueId}"
    label: str                     # canonical Neo4j label
    properties: dict
    source_record_id: str          # for provenance
    extraction_source: ExtractionSource


@dataclass
class Relationship:
    from_id: str                   # namespaced: "{dataset_id}:{uniqueId}"
    to_id: str                     # namespaced: "{dataset_id}:{uniqueId}"
    from_label: str                # Neo4j label — required for efficient MATCH
    to_label: str                  # Neo4j label — required for efficient MATCH
    type: str                      # SCREAMING_SNAKE_CASE
    properties: dict
    source_record_id: str
    extraction_source: ExtractionSource
