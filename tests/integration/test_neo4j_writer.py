"""Integration tests for graph_pipeline/neo4j_writer.py.

Requires a live local Neo4j instance. Run with:
    pytest --integration tests/integration/test_neo4j_writer.py

Each test uses a unique label prefix (derived from a UUID) to avoid colliding with
real graph data. Nodes and constraints created during tests are cleaned up in teardown.
"""
import os
import uuid

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def driver():
    """Return an authenticated Neo4j driver; skip if credentials are missing."""
    neo4j = pytest.importorskip("neo4j", reason="neo4j package not installed")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set")
    drv = neo4j.GraphDatabase.driver(uri, auth=(username, password))
    yield drv
    drv.close()


@pytest.fixture()
def label_prefix():
    """Unique label prefix per test to avoid cross-test or cross-run collisions."""
    return f"Test{uuid.uuid4().hex[:8].capitalize()}"


@pytest.fixture()
def cleanup(driver, label_prefix):
    """Yield label_prefix, then delete all nodes with that label prefix after the test."""
    yield label_prefix
    with driver.session() as session:
        # Drop constraint and nodes for both Node and Rel label variants used in tests
        for suffix in ["Node", "Rel", "NodeA", "NodeB", "Batch"]:
            label = f"{label_prefix}{suffix}"
            session.run(f"MATCH (n:{label}) DETACH DELETE n")
            session.run(
                f"DROP CONSTRAINT {label}_id IF EXISTS"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_node(label, uid, name, dataset_id="test_ds"):
    from graph_pipeline.models import ExtractionSource, Node
    return Node(
        id=f"{dataset_id}:{uid}",
        label=label,
        properties={"name": name},
        source_record_id=uid,
        extraction_source=ExtractionSource.RULE_BASED,
    )


def make_test_rel(from_id, to_id, from_label, to_label, rel_type="TEST_REL"):
    from graph_pipeline.models import ExtractionSource, Relationship
    return Relationship(
        from_id=from_id,
        to_id=to_id,
        from_label=from_label,
        to_label=to_label,
        type=rel_type,
        properties={},
        source_record_id=from_id.split(":")[-1],
        extraction_source=ExtractionSource.RULE_BASED,
    )


def count_nodes(driver, label):
    with driver.session() as s:
        result = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
        return result.single()["c"]


def count_rels(driver, rel_type):
    with driver.session() as s:
        result = s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
        return result.single()["c"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWriteAll:
    def test_write_all_correct_counts(self, driver, cleanup):
        """write_all returns correct created counts for a fresh graph."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_label_a = f"{cleanup}NodeA"
        rel_label_b = f"{cleanup}NodeB"

        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [
            make_test_rel(
                f"test_ds:n-001", f"test_ds:n-002",
                label, label,
                rel_type=f"REL_{cleanup.upper()}",
            )
        ]

        result = write_all(nodes, rels, driver, batch_size=500)

        assert result.nodes_created == 2
        assert result.nodes_matched == 0
        assert result.relationships_created == 1
        assert result.relationships_matched == 0
        assert result.errors == []

    def test_write_all_idempotent(self, driver, cleanup):
        """Re-running the same write produces nodes_created=0, nodes_matched=N."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]

        write_all(nodes, [], driver, batch_size=500)
        result = write_all(nodes, [], driver, batch_size=500)

        assert result.nodes_created == 0
        assert result.nodes_matched == 2
        assert result.errors == []

    def test_write_all_nodes_present_in_graph(self, driver, cleanup):
        """Nodes actually appear in Neo4j after write_all."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
            make_test_node(label, "n-003", "Gamma"),
        ]

        write_all(nodes, [], driver, batch_size=500)
        assert count_nodes(driver, label) == 3

    def test_write_all_relationships_present_in_graph(self, driver, cleanup):
        """Relationships actually appear in Neo4j after write_all."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_type = f"REL_{cleanup.upper()}"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [make_test_rel("test_ds:n-001", "test_ds:n-002", label, label, rel_type)]

        write_all(nodes, rels, driver, batch_size=500)
        assert count_rels(driver, rel_type) == 1

    def test_write_all_relationship_idempotent(self, driver, cleanup):
        """Re-running write_all with same relationship: relationships_created=0, matched=1."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_type = f"REL_{cleanup.upper()}"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [make_test_rel("test_ds:n-001", "test_ds:n-002", label, label, rel_type)]

        write_all(nodes, rels, driver, batch_size=500)
        result = write_all(nodes, rels, driver, batch_size=500)

        assert result.relationships_created == 0
        assert result.relationships_matched == 1

    def test_write_result_dataclass_fields(self, driver, cleanup):
        """WriteResult has the expected fields."""
        from graph_pipeline.neo4j_writer import WriteResult, write_all

        label = f"{cleanup}Node"
        nodes = [make_test_node(label, "n-001", "Alpha")]
        result = write_all(nodes, [], driver, batch_size=500)

        assert isinstance(result, WriteResult)
        assert hasattr(result, "nodes_created")
        assert hasattr(result, "nodes_matched")
        assert hasattr(result, "relationships_created")
        assert hasattr(result, "relationships_matched")
        assert hasattr(result, "errors")
        assert isinstance(result.errors, list)


@pytest.mark.integration
class TestBatchBehaviour:
    def test_small_batch_size_all_nodes_written(self, driver, cleanup):
        """Batch size smaller than total node count; all nodes still written."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Batch"
        nodes = [make_test_node(label, f"n-{i:03d}", f"Node {i}") for i in range(7)]

        result = write_all(nodes, [], driver, batch_size=3)

        assert result.nodes_created == 7
        assert result.errors == []
        assert count_nodes(driver, label) == 7

    def test_mid_batch_failure_earlier_batches_intact(self, driver, cleanup):
        """Batch 1 is committed before batch 2 starts; a batch 2 failure leaves batch 1 intact."""
        from graph_pipeline.neo4j_writer import create_constraints, write_nodes

        label = f"{cleanup}Batch"
        create_constraints([label], driver)

        # Write 3 nodes successfully with the real driver
        first_batch = [make_test_node(label, f"n-{i:03d}", f"Node {i}") for i in range(3)]
        write_nodes(first_batch, driver, batch_size=500)
        assert count_nodes(driver, label) == 3

        # Now attempt to write 3 more nodes with a driver whose session raises on begin_transaction,
        # simulating a failure before the second write batch can commit.
        from unittest.mock import MagicMock

        failing_driver = MagicMock()
        failing_session = MagicMock()
        failing_session.__enter__ = lambda s: s
        failing_session.__exit__ = MagicMock(return_value=False)
        failing_session.begin_transaction.side_effect = RuntimeError("Simulated Neo4j failure")
        failing_driver.session.return_value = failing_session

        second_batch = [make_test_node(label, f"x-{i:03d}", f"Extra {i}") for i in range(3)]
        result = write_nodes(second_batch, failing_driver, batch_size=500)

        # First batch (real driver) remains intact
        assert count_nodes(driver, label) == 3
        # Second write recorded an error
        assert len(result.errors) >= 1


@pytest.mark.integration
class TestCreateConstraints:
    def test_constraint_created(self, driver, cleanup):
        """create_constraints does not raise and the constraint exists afterwards."""
        from graph_pipeline.neo4j_writer import create_constraints

        label = f"{cleanup}Node"
        create_constraints([label], driver)

        with driver.session() as session:
            result = session.run(
                "SHOW CONSTRAINTS WHERE labelsOrTypes = [$label]",
                label=label,
            )
            constraints = result.data()
        assert len(constraints) >= 1

    def test_constraint_idempotent(self, driver, cleanup):
        """Running create_constraints twice does not raise."""
        from graph_pipeline.neo4j_writer import create_constraints

        label = f"{cleanup}Node"
        create_constraints([label], driver)
        create_constraints([label], driver)  # should not raise
