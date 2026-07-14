from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Literal

from graph_pipeline.context_store import SharedContext
from graph_pipeline.models import ExtractionSource, Node, Relationship

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    severity: Literal["error", "warning"]
    message: str
    record_id: str | None


@dataclass
class SpotCheckRecord:
    record_id: str
    node_found: bool
    relationships: list[str]  # relationship types where from_id or to_id matches this record


@dataclass
class SpotCheckReport:
    sampled: list[SpotCheckRecord]
    total_nodes: int
    total_relationships: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_referential_integrity(
    nodes: list[Node],
    relationships: list[Relationship],
    driver=None,
) -> list[ValidationError]:
    """Check that every relationship endpoint resolves to a known node.

    Step 1: check against the current batch.
    Step 2: if driver is provided, query Neo4j for any still-missing IDs.
    Step 3: IDs still unresolved → ValidationError(severity="error").
    If driver is None (dry-run), missing-from-batch → ValidationError(severity="warning").
    """
    if not relationships:
        return []

    batch_ids: set[str] = {n.id for n in nodes}
    errors: list[ValidationError] = []

    # Collect unique missing IDs across all relationships
    missing_ids: dict[str, str] = {}  # id → source_record_id of first rel that referenced it
    for rel in relationships:
        for endpoint_id in (rel.from_id, rel.to_id):
            if endpoint_id not in batch_ids and endpoint_id not in missing_ids:
                missing_ids[endpoint_id] = rel.source_record_id

    if not missing_ids:
        return []

    if driver is None:
        # Dry-run: treat all missing as warnings
        for missing_id, record_id in missing_ids.items():
            errors.append(
                ValidationError(
                    severity="warning",
                    message=f"Endpoint '{missing_id}' not found in current batch (dry-run)",
                    record_id=record_id,
                )
            )
        return errors

    # Live mode: check Neo4j for any missing IDs
    found_in_neo4j: set[str] = set()
    try:
        with driver.session() as session:
            result = session.run(
                "UNWIND $ids AS id MATCH (n {id: id}) RETURN n.id AS id",
                ids=list(missing_ids.keys()),
            )
            for record in result.data():
                found_in_neo4j.add(record["id"])
    except Exception as exc:
        logger.error("Neo4j lookup failed during referential integrity check: %s", exc)

    for missing_id, record_id in missing_ids.items():
        if missing_id not in found_in_neo4j:
            errors.append(
                ValidationError(
                    severity="error",
                    message=f"Endpoint '{missing_id}' not found in batch or in Neo4j",
                    record_id=record_id,
                )
            )

    return errors


def check_label_coverage(
    nodes: list[Node],
    shared_ctx: SharedContext,
    phantom_labels: set[str] = frozenset({"Folder"}),
) -> list[ValidationError]:
    """Warn when a node's label is not declared in the shared context.

    Labels in phantom_labels are always allowed (structural phantom nodes).
    Duplicate unknown labels produce only one warning.
    """
    known_labels: set[str] = {nt.name for nt in shared_ctx.node_types}
    known_labels |= {nt.maps_to for nt in shared_ctx.node_types}
    known_labels |= set(phantom_labels)

    errors: list[ValidationError] = []
    warned: set[str] = set()

    for node in nodes:
        if node.label not in known_labels and node.label not in warned:
            warned.add(node.label)
            errors.append(
                ValidationError(
                    severity="warning",
                    message=f"Node label '{node.label}' not found in shared context",
                    record_id=node.source_record_id or None,
                )
            )

    return errors


def spot_check(
    nodes: list[Node],
    relationships: list[Relationship],
    original_records: list[dict],
    n: int = 5,
    id_field: str = "uniqueId",
) -> SpotCheckReport:
    """Sample n records and report which nodes/relationships were extracted for each.

    Prints a human-readable summary to stdout so the human can verify counts look right.
    Does NOT assert expected counts — that's the human's job.
    """
    if not original_records:
        return SpotCheckReport(sampled=[], total_nodes=0, total_relationships=0)

    sample_size = min(n, len(original_records))
    sampled_records = random.sample(original_records, sample_size)

    node_ids: set[str] = {node.id for node in nodes}

    spot_records: list[SpotCheckRecord] = []
    for record in sampled_records:
        uid = record.get(id_field, "")
        # A node matches this record if any of its ids end with the uid (namespace-agnostic)
        node_found = any(nid.endswith(f":{uid}") for nid in node_ids)

        rel_types = [
            rel.type
            for rel in relationships
            if rel.from_id.endswith(f":{uid}") or rel.to_id.endswith(f":{uid}")
        ]

        spot_records.append(
            SpotCheckRecord(
                record_id=uid,
                node_found=node_found,
                relationships=rel_types,
            )
        )

    report = SpotCheckReport(
        sampled=spot_records,
        total_nodes=len(nodes),
        total_relationships=len(relationships),
    )

    # Print human-readable summary
    print(f"\nSpot check ({sample_size} records sampled):")
    for rec in report.sampled:
        found_mark = "✓" if rec.node_found else "✗"
        rels_str = ", ".join(rec.relationships) if rec.relationships else "none"
        print(f"  {found_mark} {rec.record_id} → relationships: {rels_str}")
    print(
        f"  Total: {report.total_nodes} nodes, {report.total_relationships} relationships\n"
    )

    return report
